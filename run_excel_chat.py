from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from excel_agent import ExcelDataAgent, AgentResponse  # noqa: E402


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
    "Show me the bottom 3 products by total cost price.",
]


def _divider(label: str = "") -> None:
    bar = "═" * 78
    print(f"\n{bar}")
    if label:
        print(f"  {label}")
        print(bar)


def _print_answer(resp: AgentResponse, turn: int, verbose: bool) -> None:
    print(f"\n  🧑‍💼 Agent [T{turn}]")
    print("  " + "─" * 74)
    for line in resp.answer.splitlines():
        print("  " + line)
    meta = [f"kind={resp.query_kind}"]
    if resp.definitions:
        meta.append(f"definitions={[d.query for d in resp.definitions]}")
    if resp.note:
        meta.append(f"note: {resp.note}")
    print(f"\n  [meta] {' | '.join(meta)}")
    if verbose:
        print("\n  📝 CODE GENERATED & EXECUTED:")
        print("  " + "─" * 74)
        for line in resp.code_used.splitlines():
            print("  │ " + line)
        if resp.code_output and resp.code_output not in resp.answer:
            print("\n  📊 RAW EXECUTION OUTPUT:")
            print("  " + "─" * 74)
            for line in resp.code_output.splitlines()[:30]:
                print("  │ " + line)
            if len(resp.code_output.splitlines()) > 30:
                print(f"  │  ... ({len(resp.code_output.splitlines()) - 30} more lines)")


def _print_welcome(agent: ExcelDataAgent, verbose: bool) -> None:
    _divider("📊 EXCEL DATA ANALYSIS AGENT — INTERACTIVE CHAT")
    print(f"  Dataset:      {agent.dataset.file_path.name}")
    print(f"  Sheet:        {agent.dataset.sheet_name}")
    print(f"  Title:        {agent.dataset.title}")
    print(f"  Rows × Cols:  {agent.dataset.row_count} × {agent.dataset.col_count}")
    print(f"  Using LLM:    {'YES' if agent._turn_counter == 0 and (Path('.env').exists() or __import__('os').environ.get('OPENAI_API_KEY') or __import__('os').environ.get('ANTHROPIC_API_KEY')) else 'No (demo rule-based)'}")
    print(f"  Verbose mode: {'ON (shows code + raw output)' if verbose else 'OFF (answers only)'}")
    print()
    print("  Features:")
    print("    ✍️  Writes & executes pandas code to query the dataset")
    print("    🔍 Looks up business/finance definitions via built-in dictionary")
    print("    📋 Summarises findings in plain English")
    print("    💬 Simple conversational chat interface")
    print()
    print("  Commands:")
    print("    quit / exit / q  →  end the chat")
    print("    summary          →  show conversation state summary")
    print("    verbose          →  toggle code/raw-output display")
    print("    sample           →  list suggested questions")
    print("    demo N           →  run first N sample questions automatically")
    print("    columns          →  show dataset column schema")
    print()
    print("  Try asking things like:")
    for q in SAMPLE_QUESTIONS[:4]:
        print(f"    • \"{q}\"")
    _divider()


def _print_summary(agent: ExcelDataAgent) -> None:
    _divider("📋 CONVERSATION STATE SUMMARY")
    print(f"  Turns taken:         {len(agent.turns)}")
    query_kinds = {}
    for t in agent.turns:
        query_kinds[t.query_class] = query_kinds.get(t.query_class, 0) + 1
    print(f"  Query type mix:      {query_kinds}")
    total_defs = sum(len(t.search_results) for t in agent.turns)
    print(f"  Definition lookups:  {total_defs}")
    print(f"  Code executions:     {agent.executor.execution_count}")
    print()
    print("  Per-turn breakdown:")
    for t in agent.turns:
        defs_q = [d.query for d in t.search_results] or "—"
        code_ok = "✅" if (t.code_result and t.code_result.success) else "❌"
        print(f"    T{t.turn_index:02d} [{t.query_class:12s}] code={code_ok}  defs={defs_q}")
    _divider()


def run_chat(excel_path: str | Path, interactive: bool = True, demo_count: int = 0, verbose: bool = False) -> None:
    agent = ExcelDataAgent(excel_path)

    _print_welcome(agent, verbose)

    if demo_count > 0:
        _divider(f"🎬 AUTO-DEMO — Running {demo_count} sample queries")
        for i, q in enumerate(SAMPLE_QUESTIONS[:demo_count], 1):
            print(f"\n  👤 User [T{i}]: {q}")
            resp = agent.ask(q)
            _print_answer(resp, i, verbose)
        _divider("END OF AUTO-DEMO")
        if not interactive:
            return
        print("  Continuing in interactive mode. Type your questions!")

    turn = len(agent.turns)
    while True:
        turn += 1
        try:
            q = input(f"\n  👤 You [T{turn}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Goodbye!")
            break
        if not q:
            turn -= 1
            continue
        ql = q.lower()
        if ql in {"quit", "exit", "q"}:
            print("  👋 Goodbye!")
            break
        if ql == "summary":
            turn -= 1
            _print_summary(agent)
            continue
        if ql == "verbose":
            turn -= 1
            verbose = not verbose
            print(f"  🔧 Verbose mode now: {'ON' if verbose else 'OFF'}")
            continue
        if ql == "sample":
            turn -= 1
            print("  💡 Suggested questions:")
            for i, sq in enumerate(SAMPLE_QUESTIONS, 1):
                print(f"    {i:2d}. {sq}")
            continue
        if ql == "columns":
            turn -= 1
            print("  🗂️  Dataset columns:")
            for c in agent.dataset.columns:
                samples = ", ".join(repr(v) for v in c.sample_values[:3])
                print(f"    • {c.clean_name:30s} [{c.dtype:18s}] — e.g. {samples}")
            continue
        if ql.startswith("demo"):
            parts = ql.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else len(SAMPLE_QUESTIONS)
            for i, sq in enumerate(SAMPLE_QUESTIONS[:n], 1):
                print(f"\n  👤 Sample [T{turn}/{i}]: {sq}")
                resp = agent.ask(sq)
                _print_answer(resp, turn, verbose)
                turn += 1
            turn -= 1
            continue

        resp = agent.ask(q)
        _print_answer(resp, turn, verbose)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_excel_chat",
        description="Excel data analysis agent — a chat interface that writes code, looks up definitions, and summarises findings.",
    )
    parser.add_argument(
        "excel",
        nargs="?",
        default=str(Path(__file__).parent / "Inventory-Records-Sample-Data.xlsx"),
        help="Path to Excel file (default: Inventory-Records-Sample-Data.xlsx in project dir)",
    )
    parser.add_argument(
        "--demo",
        type=int,
        default=0,
        metavar="N",
        help="Run first N sample queries automatically, then (optionally) continue interactive",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Exit after --demo completes (do not enter interactive chat)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show generated code and raw pandas output alongside answers",
    )
    args = parser.parse_args()

    excel_file = Path(args.excel)
    if not excel_file.exists():
        print(f"❌ Excel file not found: {excel_file}")
        sys.exit(1)

    run_chat(
        excel_path=excel_file,
        interactive=not args.non_interactive,
        demo_count=args.demo,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
