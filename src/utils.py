"""Shared helpers for baseline experiment reporting."""

import json
from pathlib import Path
from typing import Any


def save_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Save experiment results as readable, JSON-compatible output."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")
