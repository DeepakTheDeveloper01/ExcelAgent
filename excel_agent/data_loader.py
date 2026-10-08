from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.replace("\n", " ")).strip()
    name = re.sub(r"[^0-9a-zA-Z_ ]", "", name)
    name = name.strip().replace(" ", "_")
    return name or "column"


@dataclass
class ColumnInfo:
    original_name: str
    clean_name: str
    dtype: str
    sample_values: List[Any] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_name": self.original_name,
            "clean_name": self.clean_name,
            "dtype": self.dtype,
            "sample_values": self.sample_values[:5],
            "description": self.description,
        }


@dataclass
class ExcelDataset:
    file_path: Path
    sheet_name: str
    df: pd.DataFrame
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    title: str = ""

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.title or self.file_path.stem}",
            f"Sheet: {self.sheet_name}",
            f"Rows: {self.row_count}, Columns: {self.col_count}",
            "",
            "Columns:",
        ]
        for c in self.columns:
            samples = ", ".join(repr(v) for v in c.sample_values[:3])
            lines.append(f"  - {c.clean_name} ({c.dtype}) — e.g. {samples}")
        return "\n".join(lines)

    def schema_text(self) -> str:
        parts = []
        for c in self.columns:
            parts.append(
                f"Column '{c.clean_name}' (type: {c.dtype}, original: '{c.original_name}')"
                f" — sample values: {[str(v) for v in c.sample_values[:4]]}"
            )
        return "\n".join(parts)


def _find_header_row(raw: pd.DataFrame) -> int:
    for i in range(min(len(raw), 20)):
        row = raw.iloc[i].astype(str).str.lower().tolist()
        keywords = {"product", "name", "stock", "price", "cost", "units", "id", "date", "quantity", "sales", "total"}
        matches = sum(1 for cell in row if any(k in cell for k in keywords))
        if matches >= 2:
            return i
    return 0


def load_excel_dataset(path: str | Path, sheet_name: Optional[str] = None) -> ExcelDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Excel file not found: {p}")

    xl = pd.ExcelFile(p)
    sheet = sheet_name or xl.sheet_names[0]

    raw = pd.read_excel(p, sheet_name=sheet, header=None)
    header_row = _find_header_row(raw)

    df = pd.read_excel(p, sheet_name=sheet, skiprows=header_row, header=0)

    drop_cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna(how="all")

    rename_map = {}
    for col in df.columns:
        rename_map[col] = _clean_name(str(col))
    df = df.rename(columns=rename_map)

    for col in df.columns:
        if df[col].dtype == object:
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().sum() / max(1, len(df)) > 0.7:
                    df[col] = numeric
            except Exception:
                pass

    columns: List[ColumnInfo] = []
    for original, clean in rename_map.items():
        if clean not in df.columns:
            continue
        series = df[clean]
        non_null = series.dropna()
        samples = non_null.head(5).tolist() if len(non_null) > 0 else []
        dtype_str = str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            dtype_str = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype_str = "datetime"
        else:
            dtype_str = "string/categorical"

        columns.append(
            ColumnInfo(
                original_name=str(original),
                clean_name=clean,
                dtype=dtype_str,
                sample_values=samples,
            )
        )

    df = df.reset_index(drop=True)

    title_candidates = []
    raw_header_area = pd.read_excel(p, sheet_name=sheet, header=None, nrows=header_row if header_row > 0 else 5)
    for _, row in raw_header_area.iterrows():
        for v in row.tolist():
            if isinstance(v, str) and v.strip():
                title_candidates.append(v.strip())
    title = title_candidates[0] if title_candidates else p.stem

    return ExcelDataset(
        file_path=p,
        sheet_name=sheet,
        df=df,
        columns=columns,
        row_count=len(df),
        col_count=len(columns),
        title=title,
    )
