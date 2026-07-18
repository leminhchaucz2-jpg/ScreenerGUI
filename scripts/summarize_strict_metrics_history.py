"""Summarize strict-mode metrics history stored as JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_INPUT = Path("tests/artifacts/strict_mode_metrics_history.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize strict-vs-loose metrics history from a JSONL file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to JSONL history (default: {DEFAULT_INPUT.as_posix()}).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window size for recent-run stats (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary as JSON instead of plain text.",
    )
    return parser.parse_args()


def _read_history(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"History file does not exist: {path}")

    rows: list[dict[str, object]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_no} must be a JSON object.")
        rows.append(payload)

    if not rows:
        raise ValueError("History file is empty.")

    return rows


def _to_float(value: object, field: str, idx: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field '{field}' at record {idx + 1}") from exc


def _summary(rows: list[dict[str, object]], window: int) -> dict[str, object]:
    reductions = [_to_float(row.get("reduction_pct"), "reduction_pct", i) for i, row in enumerate(rows)]
    loose_counts = [_to_float(row.get("loose_count"), "loose_count", i) for i, row in enumerate(rows)]
    strict_counts = [_to_float(row.get("strict_count"), "strict_count", i) for i, row in enumerate(rows)]

    effective_window = max(1, min(window, len(rows)))
    recent_slice = rows[-effective_window:]
    recent_reductions = reductions[-effective_window:]

    latest = rows[-1]

    return {
        "records": len(rows),
        "window": effective_window,
        "overall": {
            "mean_reduction_pct": round(statistics.fmean(reductions), 4),
            "min_reduction_pct": round(min(reductions), 4),
            "max_reduction_pct": round(max(reductions), 4),
            "mean_loose_count": round(statistics.fmean(loose_counts), 4),
            "mean_strict_count": round(statistics.fmean(strict_counts), 4),
        },
        "recent": {
            "mean_reduction_pct": round(statistics.fmean(recent_reductions), 4),
            "min_reduction_pct": round(min(recent_reductions), 4),
            "max_reduction_pct": round(max(recent_reductions), 4),
            "from_record": len(rows) - effective_window + 1,
            "to_record": len(rows),
        },
        "latest": {
            "logged_at_utc": latest.get("logged_at_utc"),
            "fixture": latest.get("fixture"),
            "symbol": latest.get("symbol"),
            "timeframe": latest.get("timeframe"),
            "divergence_type": latest.get("divergence_type"),
            "indicator": latest.get("indicator"),
            "loose_count": latest.get("loose_count"),
            "strict_count": latest.get("strict_count"),
            "reduction_pct": latest.get("reduction_pct"),
        },
        "fixtures_in_recent_window": sorted(
            {
                str(row.get("fixture", "unknown"))
                for row in recent_slice
            }
        ),
    }


def _print_human(summary: dict[str, object], input_path: Path) -> None:
    overall = summary["overall"]
    recent = summary["recent"]
    latest = summary["latest"]

    print(f"History file: {input_path.as_posix()}")
    print(f"Records: {summary['records']} | Window: {summary['window']}")
    print()
    print("Overall:")
    print(f"  Mean reduction %: {overall['mean_reduction_pct']}")
    print(f"  Min reduction % : {overall['min_reduction_pct']}")
    print(f"  Max reduction % : {overall['max_reduction_pct']}")
    print(f"  Mean loose count: {overall['mean_loose_count']}")
    print(f"  Mean strict count: {overall['mean_strict_count']}")
    print()
    print("Recent Window:")
    print(f"  Mean reduction %: {recent['mean_reduction_pct']}")
    print(f"  Min reduction % : {recent['min_reduction_pct']}")
    print(f"  Max reduction % : {recent['max_reduction_pct']}")
    print(f"  Record range    : {recent['from_record']}..{recent['to_record']}")
    print()
    print("Latest Record:")
    print(f"  logged_at_utc   : {latest['logged_at_utc']}")
    print(f"  fixture         : {latest['fixture']}")
    print(f"  symbol/timeframe: {latest['symbol']} / {latest['timeframe']}")
    print(f"  setup           : {latest['divergence_type']} + {latest['indicator']}")
    print(f"  loose/strict    : {latest['loose_count']} / {latest['strict_count']}")
    print(f"  reduction_pct   : {latest['reduction_pct']}")


def main() -> int:
    args = _parse_args()
    rows = _read_history(args.input)
    summary = _summary(rows, args.window)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    else:
        _print_human(summary, args.input)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
