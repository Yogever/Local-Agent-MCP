"""JSONL metrics writer for benchmark and REPL sessions."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import TurnMetrics

LOGS_DIR = Path(__file__).parent / "logs"


class MetricsLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def log(self, metrics: TurnMetrics) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")
