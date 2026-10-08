from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


_BUILTIN_ALLOWLIST = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "int", "isinstance", "len", "list", "map", "max", "min",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "type", "zip", "print",
}

_PANDAS_ALLOWLIST = {
    "DataFrame", "Series", "pd",
}


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    code_ran: str = ""
    returned_value: Any = None
    tables: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


class CodeExecutor:
    def __init__(self, dataset_df: pd.DataFrame, df_name: str = "df"):
        self._df = dataset_df.copy()
        self._df_name = df_name
        self.execution_count = 0
        self.history: List[ExecutionResult] = []

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, pd.DataFrame):
            if len(value) == 0:
                return "(empty DataFrame)"
            return value.to_string(max_rows=30, max_cols=12)
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return "(empty Series)"
            return value.to_string(max_rows=30)
        if isinstance(value, (int, float, str, bool)):
            return str(value)
        if isinstance(value, (list, tuple)):
            lines = []
            for i, item in enumerate(value):
                lines.append(f"  [{i}] {self._format_value(item)}")
            return "\n".join(lines) if lines else "(empty)"
        if isinstance(value, dict):
            lines = []
            for k, v in value.items():
                lines.append(f"  {k}: {self._format_value(v)}")
            return "\n".join(lines) if lines else "(empty dict)"
        try:
            return str(value)
        except Exception:
            return repr(value)

    def run(self, code: str, extra_globals: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        self.execution_count += 1
        code = code.strip()

        safe_globals: Dict[str, Any] = {
            "__builtins__": {k: __builtins__[k] for k in _BUILTIN_ALLOWLIST if k in __builtins__},
            "pd": pd,
            self._df_name: self._df.copy(),
        }

        if extra_globals:
            for k, v in extra_globals.items():
                if k not in ("__builtins__",):
                    safe_globals[k] = v

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        returned: Any = None
        success = True
        error_msg = ""
        output = ""

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                compiled = compile(code, "<agent_code>", "exec")
                exec(compiled, safe_globals)

            last_line = None
            lines = code.strip().split("\n")
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and not stripped.startswith(("print ", "print(", "import ", "from ")):
                    if "=" in stripped and not any(op in stripped for op in ["==", "!=", "<=", ">="]):
                        parts = stripped.split("=", 1)
                        last_line = parts[0].strip()
                        if last_line in safe_globals:
                            returned = safe_globals[last_line]
                            break
                    else:
                        try:
                            returned = eval(stripped, safe_globals)
                        except Exception:
                            pass
                        break
        except Exception as exc:
            success = False
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            relevant = []
            for line in tb_lines:
                if "File \"<agent_code>\"" in line or not line.strip().startswith("File"):
                    relevant.append(line)
            error_msg = "".join(relevant).strip()
            if not error_msg:
                error_msg = f"{type(exc).__name__}: {exc}"

        output = stdout_buf.getvalue()
        err_output = stderr_buf.getvalue()
        if err_output:
            output = (output + "\n[stderr] " + err_output).strip()

        formatted_return = self._format_value(returned) if returned is not None else ""

        combined_output = output
        if formatted_return and formatted_return not in combined_output:
            combined_output = (combined_output + "\n" + formatted_return).strip()

        tables: List[str] = []
        if returned is not None and isinstance(returned, (pd.DataFrame, pd.Series)):
            tables.append(formatted_return)
        elif "DataFrame" in combined_output:
            tables.append(combined_output)

        result = ExecutionResult(
            success=success,
            output=combined_output,
            error=error_msg,
            code_ran=code,
            returned_value=returned,
            tables=tables,
        )
        self.history.append(result)
        return result
