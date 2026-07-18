"""Append strict-vs-loose regression metrics to a JSONL history log."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INPUT = Path("tests/artifacts/strict_mode_metrics.json")
DEFAULT_OUTPUT = Path("tests/artifacts/strict_mode_metrics_history.jsonl")


def _load_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics artifact does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Metrics artifact must be a JSON object.")

    required = {
        "fixture",
        "symbol",
        "timeframe",
        "divergence_type",
        "indicator",
        "loose_count",
        "strict_count",
        "reduction_pct",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"Metrics artifact is missing required fields: {', '.join(missing)}")

    return payload


def _append_history(metrics: dict[str, object], out_path: Path) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }

    with out_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append strict-mode regression metrics into a JSONL history file "
            "for trend tracking across runs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to metrics JSON artifact (default: {DEFAULT_INPUT.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to output JSONL history (default: {DEFAULT_OUTPUT.as_posix()}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = _load_metrics(args.input)
    record = _append_history(metrics, args.output)

    print(
        "Appended strict-mode metrics:",
        json.dumps(record, indent=2, ensure_ascii=True),
    )
    print(f"History file: {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
