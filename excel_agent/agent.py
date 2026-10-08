from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .data_loader import ExcelDataset, load_excel_dataset
from .code_executor import CodeExecutor, ExecutionResult
from .search_tool import SearchTool, SearchResult

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
USE_REAL_LLM = bool(ANTHROPIC_KEY or GROQ_KEY)


@dataclass
class AgentTurn:
    turn_index: int
    user_query: str
    query_class: str
    code_generated: str = ""
    code_result: Optional[ExecutionResult] = None
    search_results: List[SearchResult] = field(default_factory=list)
    final_answer: str = ""


@dataclass
class AgentResponse:
    answer: str
    code_used: str
    code_output: str
    tables: List[str]
    definitions: List[SearchResult]
    query_kind: str
    note: str = ""


class QueryClassifier:
    _PATTERNS: List[Tuple[str, List[str]]] = [
        ("top_n", ["top", "best", "highest", "most", "top 5", "top 10", "排名", "最高"]),
        ("bottom_n", ["bottom", "lowest", "worst", "least", "smallest", "最低"]),
        ("aggregation", ["total", "sum", "average", "mean", "median", "count", "how many", "total value", "总和", "平均", "总价值"]),
        ("comparison", ["vs", "versus", "compare", "difference between", "对比"]),
        ("percent", ["percentage", "percent", "% of", "比例"]),
        ("filter", ["which", "where", "list", "show me", "show all", "find", "筛选", "哪些"]),
        ("sort", ["sort", "order by", "rank", "排序"]),
        ("summary", ["summary", "overview", "describe the dataset", "概览", "总结数据集"]),
        ("definition", ["define", "definition of", "meaning of", "术语"]),
    ]

    _DEFINITION_STARTERS = {"what is ", "what's ", "explain "}

    @classmethod
    def classify(cls, query: str) -> str:
        q = query.lower().strip()
        q_words = set(re.findall(r"[a-z0-9_]+", q))
        stopwords = {"of", "a", "an", "the", "is", "are", "in", "on", "at", "to", "for", "and", "or", "me", "my", "i"}

        scores: Dict[str, float] = {}
        for kind, keywords in cls._PATTERNS:
            score = 0.0
            for kw in keywords:
                weight = 1.0 + 0.1 * len(kw)
                if len(kw) <= 4 or not re.match(r"^[a-z0-9 ]+$", kw):
                    if not re.match(r"^[a-z0-9 ]+$", kw):
                        if kw in q:
                            padded = " " + q + " "
                            padded_kw = " " + kw + " "
                            if padded_kw in padded or q.startswith(kw + " ") or q.endswith(" " + kw):
                                score += weight
                        continue
                    kw_tokens = set(t for t in re.findall(r"[a-z0-9_]+", kw) if t and t not in stopwords)
                    if kw_tokens:
                        query_content = q_words - stopwords
                        if kw_tokens.issubset(query_content):
                            score += weight
                            continue
                    padded = " " + q + " "
                    padded_kw = " " + kw + " "
                    if padded_kw in padded or q.startswith(kw + " ") or q.endswith(" " + kw):
                        score += weight
                else:
                    if kw in q:
                        score += weight
            if score:
                scores[kind] = score
        if re.search(r"\b(top|bottom)\s*\d+", q):
            m = re.search(r"\b(top|bottom)\s*\d+", q)
            if m.group(1) == "top":
                scores["top_n"] = scores.get("top_n", 0) + 2
            else:
                scores["bottom_n"] = scores.get("bottom_n", 0) + 2
        if re.search(r"\b(vs|versus|compare)\b", q) or " vs " in q:
            scores["comparison"] = scores.get("comparison", 0) + 2

        data_kinds = {"top_n", "bottom_n", "aggregation", "comparison", "filter", "sort", "percent", "summary"}
        data_scores = {k: v for k, v in scores.items() if k in data_kinds}
        definition_score = scores.get("definition", 0)

        starts_like_definition = any(q.startswith(s) for s in cls._DEFINITION_STARTERS)

        if data_scores:
            sorted_kinds = sorted(data_scores.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
            best_data_kind, best_data_score = sorted_kinds[0]
            if best_data_score >= definition_score:
                return best_data_kind

        if definition_score > 0:
            return "definition"

        if starts_like_definition and len(q.split()) <= 8:
            data_toks = {"total", "sum", "average", "list", "show", "top", "many", "units", "sold"}
            if not any(tok in q_words for tok in data_toks):
                return "definition"

        if not scores:
            if any(ch.isdigit() for ch in q) or "more than" in q or "less than" in q or "greater than" in q:
                return "filter"
            if any(tok in q_words for tok in {"dataset", "data", "inventory", "product"}):
                return "summary"
            return "general"

        sorted_all = sorted(scores.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
        return sorted_all[0][0]


class _CodeStrategist:
    def __init__(self, dataset: ExcelDataset):
        self.ds = dataset
        self.col_map = {c.clean_name: c for c in dataset.columns}

    @staticmethod
    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for c in seq:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def numeric_cols(self) -> List[str]:
        return [c.clean_name for c in self.ds.columns if c.dtype == "numeric"]

    def string_cols(self) -> List[str]:
        return [c.clean_name for c in self.ds.columns if c.dtype == "string/categorical"]

    def _pick_col(self, keywords: List[str], candidates: List[str]) -> Optional[str]:
        q_tokens = [t.lower() for k in keywords for t in re.split(r"\W+", k) if t]
        best_col = None
        best_score = 0
        for c in candidates:
            col_tokens = set(re.split(r"_+", c.lower()))
            original_tokens = set(re.split(r"\W+", self.col_map[c].original_name.lower()))
            all_tokens = col_tokens | original_tokens
            score = sum(1 for t in q_tokens if t in all_tokens)
            partial = sum(1 for t in q_tokens if any(t in tok for tok in all_tokens))
            combined = score * 2 + partial
            if combined > best_score:
                best_score = combined
                best_col = c
        return best_col if best_score >= 1 else None

    def _match_value(self, query: str, string_col: str) -> Optional[str]:
        vals = self.ds.df[string_col].dropna().astype(str).tolist()
        q_lower = query.lower()
        for v in sorted(vals, key=len, reverse=True):
            if v and v.lower() in q_lower:
                return v
        for v in vals:
            vnorm = re.sub(r"[^a-z0-9]", "", v.lower())
            qnorm = re.sub(r"[^a-z0-9]", "", q_lower)
            if vnorm and vnorm in qnorm and len(vnorm) >= 3:
                return v
        return None

    def _extract_number(self, query: str) -> Optional[Tuple[float, str]]:
        m = re.search(r"(more than|greater than|above|over|>\s*)?(\d+(?:\.\d+)?)", query.lower())
        if not m:
            return None
        direction = "gt" if m.group(1) else "eq"
        m2 = re.search(r"(less than|below|under|<\s*)\s*(\d+(?:\.\d+)?)", query.lower())
        if m2:
            return (float(m2.group(2)), "lt")
        return (float(m.group(2)), direction)

    def generate(self, query: str, kind: str) -> str:
        q = query
        ql = q.lower()
        num_cols = self.numeric_cols()
        str_cols = self.string_cols()

        measure_col = self._pick_col(
            ["cost", "price", "total", "value", "stock", "sold", "units", "revenue"],
            num_cols,
        ) or (num_cols[-1] if num_cols else "")
        name_col = self._pick_col(["name", "product", "item"], str_cols) or (str_cols[0] if str_cols else "")
        id_col = self._pick_col(["id", "code"], str_cols)
        stock_col = self._pick_col(["hand", "closing", "stock"], num_cols) or measure_col
        sold_col = self._pick_col(["sold", "sales"], num_cols) or measure_col
        price_col = self._pick_col(["per unit", "unit price", "price per"], num_cols)

        if kind == "summary":
            return (
                "desc = df.describe(include='all').round(2)\n"
                "print('=== Dataset Overview ===')\n"
                f"print(f'Rows: {{len(df)}}, Columns: {{len(df.columns)}}')\n"
                "print('\\n=== Numeric summary ===')\n"
                "print(desc)\n"
                + (f"print('\\n=== Total inventory value ===')\n"
                   f"total_value = (df['{stock_col}'] * df['{price_col}']).sum() if '{price_col}' in df.columns else df['{stock_col}'].sum()\n"
                   f"print(f'$ {{total_value:,.2f}}')\n"
                   f"print(f'\\n=== Top 5 by {measure_col} ===')\n"
                   f"print(df.sort_values('{measure_col}', ascending=False).head(5)[['{name_col}', '{measure_col}']].to_string())\n"
                   if measure_col and name_col else "")
            )

        if kind == "top_n":
            n_match = re.search(r"top\s*(\d+)", ql)
            n = int(n_match.group(1)) if n_match else 5
            sort_col = measure_col
            if "stock" in ql or "on hand" in ql or "inventory" in ql:
                sort_col = stock_col
            if "sold" in ql or "sales" in ql or "best selling" in ql:
                sort_col = sold_col
            disp_cols = self._dedupe([c for c in [name_col, id_col, sort_col, stock_col, sold_col] if c and c in self.ds.df.columns])
            disp = ", ".join(f"'{c}'" for c in disp_cols[:5]) or "*"
            return (
                f"result = df.sort_values('{sort_col}', ascending=False).head({n})[[{disp}]]\n"
                f"print(f'Top {n} products by {sort_col}:')\n"
                "print(result.to_string(index=False))\n"
                "result"
            )

        if kind == "bottom_n":
            n_match = re.search(r"bottom\s*(\d+)", ql)
            n = int(n_match.group(1)) if n_match else 5
            sort_col = measure_col
            if "stock" in ql or "on hand" in ql:
                sort_col = stock_col
            if "sold" in ql:
                sort_col = sold_col
            disp_cols = self._dedupe([c for c in [name_col, id_col, sort_col, stock_col, sold_col] if c and c in self.ds.df.columns])
            disp = ", ".join(f"'{c}'" for c in disp_cols[:5]) or "*"
            return (
                f"result = df.sort_values('{sort_col}', ascending=True).head({n})[[{disp}]]\n"
                f"print(f'Bottom {n} products by {sort_col}:')\n"
                "print(result.to_string(index=False))\n"
                "result"
            )

        if kind == "aggregation":
            parts: List[str] = []
            target_col = measure_col
            if "count" in ql or "how many" in ql or "number of" in ql:
                op_word = "count"
                if "product" in ql:
                    parts.append(f"print(f'Total number of products: {{len(df)}}')\n")
                    parts.append("len(df)")
                else:
                    parts.append(f"count_result = df['{target_col}'].count()\n")
                    parts.append(f"print(f'Count of {{\"{target_col}\"}}: {{count_result}}')\n")
                    parts.append("count_result")
                return "".join(parts)

            if "average" in ql or "mean" in ql or "avg" in ql:
                if "per unit" in ql or "unit price" in ql or "price per" in ql:
                    target_col = price_col or measure_col
                elif "stock" in ql or "inventory" in ql:
                    target_col = stock_col
                elif "sold" in ql or "sales" in ql:
                    target_col = sold_col
                else:
                    target_col = measure_col
                parts.append(f"avg_val = df['{target_col}'].mean()\n")
                parts.append(f"print(f'Average {target_col}: {{avg_val:,.2f}}')\n")
                parts.append("avg_val")
                return "".join(parts)

            if "median" in ql:
                parts.append(f"med_val = df['{target_col}'].median()\n")
                parts.append(f"print(f'Median {target_col}: {{med_val:,.2f}}')\n")
                parts.append("med_val")
                return "".join(parts)

            agg_col = measure_col
            if "value" in ql or "total cost" in ql or "worth" in ql:
                if price_col and stock_col:
                    parts.append(
                        f"total_inv_value = (df['{stock_col}'] * df['{price_col}']).sum()\n"
                        f"print(f'Total inventory value (stock × unit cost): $ {{total_inv_value:,.2f}}')\n"
                    )
                    parts.append("total_inv_value")
                    return "".join(parts)
            if "sold" in ql or "sales" in ql:
                if price_col and sold_col:
                    parts.append(
                        f"total_sold_value = (df['{sold_col}'] * df['{price_col}']).sum()\n"
                        f"print(f'Total cost value of units sold: $ {{total_sold_value:,.2f}}')\n"
                    )
                    parts.append("total_sold_value")
                    return "".join(parts)

            parts.append(f"total_val = df['{agg_col}'].sum()\n")
            parts.append(f"print(f'Sum of {agg_col}: {{total_val:,.2f}}')\n")
            parts.append("total_val")
            return "".join(parts)

        if kind == "percent":
            num_info = self._extract_number(q)
            threshold = num_info[0] if num_info else None
            if threshold is not None:
                parts = [
                    f"count_above = (df['{stock_col}'] > {threshold}).sum()\n",
                    f"pct = count_above / len(df) * 100\n",
                    f"print(f'Products with {stock_col} > {threshold}: {{count_above}} / {{len(df)}} ({{pct:.1f}}%)')\n",
                    "pct"
                ]
                return "".join(parts)
            if price_col and stock_col:
                parts = [
                    "total_value = (df['{}'] * df['{}']).sum()\n".format(stock_col, price_col),
                    f"ranked = df.assign(value = df['{stock_col}'] * df['{price_col}']).sort_values('value', ascending=False)\n",
                    "ranked['pct_of_total'] = ranked['value'] / ranked['value'].sum() * 100\n",
                    "top = ranked.head(5)[['{}', 'value', 'pct_of_total']]\n".format(name_col),
                    "print(top.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))\n",
                    "top"
                ]
                return "".join(parts)
            return f"print(df.columns.tolist())\ndf.describe()"

        if kind == "comparison":
            vals_to_compare: List[str] = []
            for col in str_cols:
                for v in self.ds.df[col].dropna().astype(str).unique():
                    if v.lower() in ql and len(v) >= 3 and v not in vals_to_compare:
                        vals_to_compare.append(v)
            if len(vals_to_compare) >= 2:
                targets = vals_to_compare[:2]
                disp_cols = self._dedupe([c for c in [name_col, stock_col, sold_col, measure_col] if c])
                cols_str = ", ".join(f"'{c}'" for c in disp_cols)
                return (
                    f"mask = df['{name_col}'].isin({targets!r})\n"
                    f"compared = df.loc[mask, [{cols_str}]]\n"
                    f"print(f'Comparison of {targets[0]} vs {targets[1]}:')\n"
                    "print(compared.to_string(index=False))\n"
                    "compared"
                )
            return f"print('Available categories:', df['{name_col}'].nunique(), 'unique values in {name_col}')\ndf.head(10)"

        if kind == "filter" or kind == "sort":
            num_info = self._extract_number(q)
            matched_name = self._match_value(q, name_col) if name_col else None
            if matched_name:
                disp_cols = self._dedupe([c for c in [name_col, id_col, stock_col, sold_col, measure_col, price_col] if c])
                cols_str = ", ".join(f"'{c}'" for c in disp_cols)
                product_value = matched_name.replace("'", "\\'")
                product_lower_repr = repr(matched_name.lower())
                return (
                    f"row = df[df['{name_col}'].astype(str).str.lower() == {product_lower_repr}]\n"
                    f"print('Details for product: {product_value}')\n"
                    "print(row.to_string(index=False))\n"
                    "row"
                )
            if num_info is not None:
                val, direction = num_info
                filter_col = stock_col
                if "sold" in ql:
                    filter_col = sold_col
                if "price" in ql:
                    filter_col = price_col or measure_col
                if "value" in ql and price_col and stock_col:
                    op = ">" if direction in ("gt", "eq") else "<"
                    return (
                        f"mask = (df['{stock_col}'] * df['{price_col}']) {op} {val}\n"
                        f"result = df.loc[mask].assign(_value = df.loc[mask, '{stock_col}'] * df.loc[mask, '{price_col}'])\n"
                        "result = result.sort_values('_value', ascending=False).head(20)\n"
                        f"cols = ['{name_col}', '{stock_col}', '{price_col}', '_value']\n"
                        "print(f'Products where total value {op} {val}: {{len(result)}} found')\n"
                        "print(result[cols].to_string(index=False))\n"
                        "result"
                    )
                op = ">" if direction in ("gt", "eq") else "<"
                disp_cols = self._dedupe([c for c in [name_col, id_col, filter_col, stock_col, sold_col] if c])
                cols_str = ", ".join(f"'{c}'" for c in disp_cols)
                return (
                    f"mask = df['{filter_col}'] {op} {val}\n"
                    f"result = df.loc[mask, [{cols_str}]].sort_values('{filter_col}', ascending=False)\n"
                    f"print(f'Products where {filter_col} {op} {val}: {{len(result)}} found (top 20 shown):')\n"
                    "print(result.head(20).to_string(index=False))\n"
                    "result.head(20)"
                )
            disp_cols = self._dedupe([c for c in [name_col, id_col, stock_col, sold_col, measure_col] if c])
            cols_str = ", ".join(f"'{c}'" for c in disp_cols)
            sort_dir = "False" if kind == "sort" and ("ascending" in ql or "low to high" in ql) else "False"
            sc = measure_col if "price" in ql or "total" in ql else stock_col
            return (
                f"result = df.sort_values('{sc}', ascending={sort_dir}).head(20)[[{cols_str}]]\n"
                f"print(f'Sorted inventory (first 20 by {sc}):')\n"
                "print(result.to_string(index=False))\n"
                "result"
            )

        if kind == "definition":
            term = SearchTool.detect_definition_need(q) or [q.strip("?")]
            return f"# This query needs a definition lookup, not data code.\n# Query term: {term[0]}\nprint('Looking up definition for: {term[0]}')\nNone"

        return (
            "result = df.describe(include='all').round(2)\n"
            "print('Dataset summary:')\n"
            "print(result)\n"
            "result"
        )


def _call_llm(prompt: str, max_tokens: int = 1200) -> str:
    errors: List[str] = []
    if ANTHROPIC_KEY:
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [getattr(b, "text", "") for b in getattr(msg, "content", []) if getattr(b, "type", "") == "text"]
            response = "\n".join(parts).strip()
            if response:
                return response
            errors.append("Anthropic returned an empty response")
        except Exception as exc:
            errors.append(f"Anthropic: {exc}")
    if GROQ_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
            resp = client.chat.completions.create(model="openai/gpt-oss-120b",messages=[
            {"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.2
            )
            response = (resp.choices[0].message.content or "").strip()
            if response:
                return response
            errors.append("Groq returned an empty response")
        except Exception as exc:
            errors.append(f"Groq: {exc}")
    if errors:
        return "[LLM error: " + "; ".join(errors) + "]"
    return ""


def _extract_python_code(text: str) -> str:
    block_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if block_match:
        return block_match.group(1).strip()
    indented = re.findall(r"^(?: {4,}|\t).+$", text, re.MULTILINE)
    if indented:
        return "\n".join(line.strip() for line in indented).strip()
    return text.strip()


class ExcelDataAgent:
    def __init__(self, excel_path: str | Path):
        self.dataset = load_excel_dataset(excel_path)
        self.executor = CodeExecutor(self.dataset.df)
        self.search = SearchTool()
        self.strategist = _CodeStrategist(self.dataset)
        self.turns: List[AgentTurn] = []
        self._turn_counter = 0
        self._llm_summary_warning_shown = False

    def _needs_definition_lookup(self, query: str, kind: str) -> List[str]:
        if kind == "definition":
            terms = SearchTool.detect_definition_need(query)
            if not terms:
                cleaned = query.strip().rstrip("?.!").strip()
                cleaned = re.sub(r"^(what|define|explain)\s+(is|are|a|an|the)?\s*", "", cleaned, flags=re.IGNORECASE)
                terms = [cleaned]
            return terms
        terms = SearchTool.detect_definition_need(query)
        if terms:
            return terms
        return []

    def _generate_code_llm(self, query: str, kind: str) -> str:
        schema = self.dataset.schema_text()
        prompt = f"""You are a data analyst assistant that writes Python pandas code to answer business questions.

Dataset title: {self.dataset.title}
Columns (clean_name and original_name, with type and samples):
{schema}

The DataFrame is already loaded in variable `df`.
Write ONLY Python pandas code to answer this question. Put print statements for human-readable output and assign the final result to a variable (or let the last expression be a DataFrame/value).

USER QUESTION: {query}
QUERY TYPE: {kind}

Respond with the code inside a single ```python ... ``` block.
"""
        raw = _call_llm(prompt, max_tokens=800)
        if raw.startswith("[LLM error") or not raw:
            return self.strategist.generate(query, kind)
        return _extract_python_code(raw) or self.strategist.generate(query, kind)

    def _summarize_llm(self, query: str, kind: str, code_output: str, definitions: List[SearchResult]) -> str:
        schema = self.dataset.schema_text()
        defs_text = "\n".join(
            f"DEFINITION of '{d.query}': {d.definition}" + (f" [formula: {d.formula}]" if d.formula else "")
            for d in definitions
        )
        prompt = f"""You are a friendly, concise business-data assistant summarising results in plain English.

Dataset: {self.dataset.title} ({self.dataset.row_count} rows, {self.dataset.col_count} cols)
Original user question: {query}
Query category: {kind}

Code execution output (what pandas returned / printed):
```
{code_output[:2500]}
```

{defs_text}

Summarise the findings in 3-6 sentences of plain, clear English. Mention key numbers, rankings, and totals. If a definition lookup was performed, weave it in naturally. End with 2-3 bullet 'Key findings' points. No technical jargon unless explained. Do NOT describe the code or the process.
"""
        raw = _call_llm(prompt, max_tokens=700)
        return raw.strip()

    def _summarize_demo(self, query: str, kind: str, result: ExecutionResult, definitions: List[SearchResult]) -> str:
        output = result.output.strip() or "(no output)"
        lines = output.splitlines()[:8]

        intro = f"For your question about {query.strip()}, here are the findings from the dataset ({self.dataset.title}):"

        if definitions:
            d = definitions[0]
            intro = f"📖 **{d.query.upper()}** — {d.definition}  \n\n" + intro
            if d.formula:
                intro += f"  Formula: {d.formula}\n\n"

        parts: List[str] = [intro, ""]
        parts.append("```")
        parts.extend(lines)
        parts.append("```")

        key_points: List[str] = []
        numbers = re.findall(r"(?<![A-Za-z])\$?\s?\d[\d,]*\.?\d*", output)
        for num in numbers[:3]:
            key_points.append(f"Notable value found: {num}")

        if self.strategist.string_cols():
            name_col = self.strategist._pick_col(["name", "product"], self.strategist.string_cols())
            if name_col and result.returned_value is not None:
                try:
                    import pandas as _pd
                    rv = result.returned_value
                    if isinstance(rv, _pd.DataFrame) and name_col in rv.columns:
                        top_names = rv[name_col].astype(str).head(3).tolist()
                        if top_names:
                            key_points.append("Top items: " + ", ".join(top_names))
                except Exception:
                    pass

        if key_points:
            parts.append("")
            parts.append("Key findings:")
            for kp in key_points:
                parts.append(f"  • {kp}")
        else:
            parts.append("")
            parts.append("Key findings:")
            parts.append(f"  • Query type: {kind} — use the raw output above to draw conclusions.")
            parts.append(f"  • Dataset has {self.dataset.row_count} records across {self.dataset.col_count} columns.")

        return "\n".join(parts)

    def ask(self, user_query: str) -> AgentResponse:
        self._turn_counter += 1
        kind = QueryClassifier.classify(user_query)

        def_terms = self._needs_definition_lookup(user_query, kind)
        search_results: List[SearchResult] = []
        for term in def_terms:
            sr = self.search.lookup(term)
            if sr:
                search_results.append(sr)

        if USE_REAL_LLM:
            code = self._generate_code_llm(user_query, kind)
        else:
            code = self.strategist.generate(user_query, kind)

        exec_result = self.executor.run(code)
        if not exec_result and USE_REAL_LLM:
            retry_code = self.strategist.generate(user_query, kind)
            exec_result = self.executor.run(retry_code)
            code = code + "\n# (fallback rule-based code)\n" + retry_code

        combined_output = exec_result.output
        if not exec_result.success and exec_result.error:
            combined_output = f"[Code execution failed] {exec_result.error}\n\n{combined_output}"

        note_parts: List[str] = []
        if USE_REAL_LLM:
            summary = self._summarize_llm(user_query, kind, combined_output, search_results)
            if summary.startswith("[LLM error") or not summary:
                llm_error = summary
                summary = self._summarize_demo(user_query, kind, exec_result, search_results)
                if not self._llm_summary_warning_shown:
                    detail = f" Details: {llm_error}" if llm_error else " The LLM returned an empty response."
                    note_parts.append("LLM summarization failed; using built-in summary." + detail)
                    self._llm_summary_warning_shown = True
        else:
            summary = self._summarize_demo(user_query, kind, exec_result, search_results)
            note_parts.append("Demo mode (no LLM configured). Set GROQ_API_KEY or ANTHROPIC_API_KEY for smarter answers.")

        if search_results:
            note_parts.append(f"Definitions looked up: {[r.query for r in search_results]}")
        if not exec_result.success:
            note_parts.append("Code execution did not fully succeed.")

        tables = exec_result.tables or ([combined_output] if combined_output else [])

        turn = AgentTurn(
            turn_index=self._turn_counter,
            user_query=user_query,
            query_class=kind,
            code_generated=code,
            code_result=exec_result,
            search_results=search_results,
            final_answer=summary,
        )
        self.turns.append(turn)

        return AgentResponse(
            answer=summary,
            code_used=code,
            code_output=combined_output,
            tables=tables,
            definitions=search_results,
            query_kind=kind,
            note="; ".join(note_parts),
        )

    def reset(self) -> None:
        self._turn_counter = 0
        self.turns = []
        self.executor = CodeExecutor(self.dataset.df)
