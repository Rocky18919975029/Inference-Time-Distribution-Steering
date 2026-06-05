from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval_outputs.py"
SPEC = importlib.util.spec_from_file_location("compare_eval_outputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_and_classify_rows(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    rows = [
        {"idx": 1, "score": [True], "finish_reason": ["stop"]},
        {"idx": 2, "score": [False], "finish_reason": ["invalid_eval_row"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = MODULE._load_rows(path)

    assert MODULE._is_correct(loaded["1"])
    assert not MODULE._is_correct(loaded["2"])
    assert MODULE._is_invalid(loaded["2"])


def test_format_example_includes_both_responses() -> None:
    base = {"question": "Q", "gt": "A", "pred": ["B"], "code": ["base response"]}
    itds = {"question": "Q", "gt": "A", "pred": ["A"], "code": ["itds response"]}

    rendered = MODULE._format_example("7", base, itds)

    assert "base response" in rendered
    assert "itds response" in rendered
    assert "base_pred: B" in rendered
    assert "itds_pred: A" in rendered
