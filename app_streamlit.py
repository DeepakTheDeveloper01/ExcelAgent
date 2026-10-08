from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from excel_agent import ExcelDataAgent, AgentResponse  # noqa: E402
from excel_agent.data_loader import ExcelDataset, load_excel_dataset  # noqa: E402


SAMPLE_QUESTIONS = [
    "What is the total inventory value of all products?",
    "Show me the top 5 products by number of units sold.",
    "Which products have more than 70 units on hand?",
    "What is the average cost price per unit?",
    "Find details for product 'Laptop'.",
    "Define inventory turnover.",
    "Compare Laptop vs Monitor.",
    "Give me a summary of the dataset.",
    "What percentage of products have more than 50 units on hand?",
    "Bottom 3 products by total cost price.",
    "Define safety stock.",
    "What is COGS?",
]

QUERY_KIND_EMOJI = {
    "aggregation": "🧮",
    "top_n": "🏆",
    "bottom_n": "🔻",
    "filter": "🔍",
    "comparison": "⚖️",
    "percent": "📊",
    "summary": "📋",
    "sort": "↕️",
    "definition": "📖",
    "general": "💬",
}


DEFAULT_EXCEL = Path(__file__).parent / "Inventory-Records-Sample-Data.xlsx"


def _render_dataset_card(ds: ExcelDataset) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 File", ds.file_path.name)
    col2.metric("📄 Sheet", ds.sheet_name)
    col3.metric("🗂️  Rows", f"{ds.row_count:,}")
    col4.metric("📊 Columns", f"{ds.col_count:,}")

    with st.expander(f"🗂️  Column schema ({len(ds.columns)} columns)", expanded=False):
        rows = []
        for c in ds.columns:
            rows.append(
                {
                    "Column": c.clean_name,
                    "Original header": c.original_name,
                    "Type": c.dtype,
                    "Sample values": ", ".join(repr(v) for v in c.sample_values[:4]),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("👀 First 10 rows of the dataset", expanded=False):
        st.dataframe(ds.df.head(10), use_container_width=True, hide_index=True)


def _render_answer(resp: AgentResponse, show_code: bool, show_output: bool) -> None:
    emoji = QUERY_KIND_EMOJI.get(resp.query_kind, "💬")
    st.caption(f"{emoji}  Query type: **{resp.query_kind}**")

    for line in resp.answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if stripped.startswith("📖 **"):
            st.info(stripped)
            continue
        if stripped.startswith("Key findings:"):
            st.markdown("**Key findings**")
            continue
        if stripped.startswith("• ") or stripped.startswith("- "):
            st.markdown(f"- {stripped[2:]}")
            continue
        if stripped == "":
            st.write("")
            continue
        st.write(stripped)

    if resp.definitions:
        with st.expander("📖 Definitions looked up", expanded=True):
            for d in resp.definitions:
                st.markdown(f"**{d.query}** — *{d.category}*")
                st.write(d.definition)
                if d.formula:
                    st.code(d.formula, language="text")
                if d.note:
                    st.caption(d.note)

    if show_code and resp.code_used.strip():
        with st.expander("✍️  Pandas code generated & executed", expanded=False):
            st.code(resp.code_used, language="python")

    if show_output and resp.code_output.strip():
        with st.expander("📊 Raw execution output", expanded=False):
            st.text(resp.code_output[:6000])

    if resp.note:
        st.caption("ℹ️  " + resp.note)


def _build_agent(excel_path: Path) -> ExcelDataAgent:
    return ExcelDataAgent(excel_path)


def main() -> None:
    st.set_page_config(
        page_title="Excel Data Analysis Agent",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Excel Data Analysis Agent")
    st.caption(
        "Ask natural-language questions about any Excel dataset. "
        "The agent writes pandas code, looks up definitions, and summarises findings in plain English."
    )

    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {
            "excel_path": None,
            "dataset": None,
            "agent": None,
            "messages": [],
        }

    with st.sidebar:
        st.header("⚙️  Settings")

        st.subheader("1. Choose your Excel file")
        source = st.radio(
            "Source",
            ["Sample inventory dataset", "Upload an Excel file"],
            label_visibility="collapsed",
        )
        temp_path: Optional[Path] = None
        if source == "Upload an Excel file":
            uploaded = st.file_uploader(
                "Upload .xlsx / .xls", type=["xlsx", "xls"], accept_multiple_files=False
            )
            if uploaded is not None:
                suffix = Path(uploaded.name).suffix or ".xlsx"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.getbuffer())
                tmp.flush()
                tmp.close()
                temp_path = Path(tmp.name)
                chosen_path = temp_path
            else:
                chosen_path = DEFAULT_EXCEL
        else:
            chosen_path = DEFAULT_EXCEL

        if (
            st.session_state.agent_state["excel_path"] != chosen_path
            or st.session_state.agent_state["dataset"] is None
        ):
            try:
                ds = load_excel_dataset(chosen_path)
                st.session_state.agent_state["excel_path"] = chosen_path
                st.session_state.agent_state["dataset"] = ds
                st.session_state.agent_state["agent"] = _build_agent(chosen_path)
                st.session_state.agent_state["messages"] = []
            except Exception as exc:
                st.error(f"Failed to load Excel file: {exc}")
                st.stop()

        st.subheader("2. Display options")
        show_code = st.checkbox("Show generated pandas code", value=False)
        show_output = st.checkbox("Show raw execution output", value=False)

        st.divider()
        if st.button("🧹 Reset chat", use_container_width=True):
            agent = st.session_state.agent_state["agent"]
            if agent is not None:
                agent.reset()
            st.session_state.agent_state["messages"] = []
            st.rerun()

    state = st.session_state.agent_state
    ds: Optional[ExcelDataset] = state["dataset"]
    agent: Optional[ExcelDataAgent] = state["agent"]
    if ds is None or agent is None:
        st.info("👈 Select an Excel file on the sidebar to begin.")
        return

    with st.container(border=True):
        st.subheader(f"📂 Dataset: {ds.title}")
        _render_dataset_card(ds)

    st.divider()
    st.subheader("💬 Chat")

    with st.expander("💡 Suggested questions (click to copy, or use buttons below)", expanded=False):
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            st.markdown(f"{i}. {q}")

    st.caption("Quick-start buttons:")
    cols = st.columns(3)
    button_questions = SAMPLE_QUESTIONS[:6]
    pressed_q: Optional[str] = None
    for idx, q in enumerate(button_questions):
        col = cols[idx % len(cols)]
        short = q if len(q) < 45 else q[:43] + "…"
        if col.button(short, key=f"sq_{idx}", use_container_width=True, help=q):
            pressed_q = q

    st.write("")

    for msg in state["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(content)
        else:
            with st.chat_message("assistant", avatar="🧑‍💼"):
                _render_answer(content, show_code, show_output)

    user_q = st.chat_input("Ask a question about the dataset… (e.g. 'Which products have more than 50 units on hand?')")
    if pressed_q:
        user_q = pressed_q

    if user_q:
        state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user", avatar="👤"):
            st.write(user_q)

        with st.chat_message("assistant", avatar="🧑‍💼"):
            with st.status("Thinking… classifying query", expanded=False) as status:
                try:
                    from excel_agent.agent import QueryClassifier
                    kind = QueryClassifier.classify(user_q)
                    status.update(
                        label=f"Query classified as **{kind}** {QUERY_KIND_EMOJI.get(kind, '')}. Generating pandas code…",
                    )
                    resp = agent.ask(user_q)
                    status.update(
                        label=f"✅ Answer ready ({resp.query_kind}, {len(resp.definitions)} definition lookups)",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:
                    status.update(label=f"❌ Error: {exc}", state="error", expanded=True)
                    resp = AgentResponse(
                        answer=f"Sorry, I hit an error while answering: {exc}",
                        code_used="",
                        code_output="",
                        tables=[],
                        definitions=[],
                        query_kind="general",
                        note=f"Exception: {type(exc).__name__}",
                    )
            _render_answer(resp, show_code, show_output)

        state["messages"].append({"role": "assistant", "content": resp})

    st.divider()
    turns = len(agent.turns)
    if turns > 0:
        with st.expander(f"📜 Conversation state ({turns} turns so far)", expanded=False):
            breakdown_rows = []
            for t in agent.turns:
                code_ok = t.code_result.success if t.code_result else False
                breakdown_rows.append(
                    {
                        "Turn": t.turn_index,
                        "Query type": t.query_class,
                        "Code OK": "✅" if code_ok else "❌",
                        "Definitions looked up": ", ".join(d.query for d in t.search_results) or "—",
                        "Question": (t.user_query[:90] + "…") if len(t.user_query) > 90 else t.user_query,
                    }
                )
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)
            st.metric("Total code executions", agent.executor.execution_count)
            total_defs = sum(len(t.search_results) for t in agent.turns)
            st.metric("Total definition lookups", total_defs)


if __name__ == "__main__":
    main()
