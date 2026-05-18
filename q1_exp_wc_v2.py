#!/usr/bin/env python3

import json
import csv
import argparse
import re
from pathlib import Path
from typing import Any, Optional, Dict


QUESTION_1_RE = re.compile(r"^\s*Question\s*1\s*:", re.IGNORECASE)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_parse_json_string(value: Any) -> Any:
    """
    Some inputs have fields like explanation that are JSON-encoded strings
    (e.g., "[{...}]"). If so, parse them. Otherwise return as-is.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return value
    return value


def _extract_explanation_from_question_obj(qobj: Any) -> Optional[str]:
    """
    Given a question object (dict), return explanation text.
    Handles:
      - qobj["explanation"]
      - qobj["model_json"]["explanation"]
    And cases where explanation is a JSON-encoded string representing a list/dict.
    """
    if not isinstance(qobj, dict):
        return None

    # Prefer top-level explanation
    expl = qobj.get("explanation")
    expl = _maybe_parse_json_string(expl)

    # If explanation parsed into list/dict, try to extract explanation field(s)
    if isinstance(expl, list) and expl:
        # e.g. [{"letter": "D", "answer": "...", "explanation": "..."}]
        first = expl[0]
        if isinstance(first, dict) and isinstance(first.get("explanation"), str):
            return first["explanation"]
        # fallback: stringify
        return " ".join(str(x) for x in expl if x is not None)

    if isinstance(expl, dict):
        # e.g. {"explanation": "..."} or nested shapes
        inner = expl.get("explanation")
        if isinstance(inner, str):
            return inner
        return json.dumps(expl, ensure_ascii=False)

    if isinstance(expl, str) and expl.strip():
        return expl

    # Fallback to model_json.explanation
    mj = qobj.get("model_json")
    mj = _maybe_parse_json_string(mj)
    if isinstance(mj, dict):
        mj_expl = mj.get("explanation")
        mj_expl = _maybe_parse_json_string(mj_expl)

        if isinstance(mj_expl, list) and mj_expl:
            first = mj_expl[0]
            if isinstance(first, dict) and isinstance(first.get("explanation"), str):
                return first["explanation"]
            return " ".join(str(x) for x in mj_expl if x is not None)

        if isinstance(mj_expl, dict):
            inner = mj_expl.get("explanation")
            if isinstance(inner, str):
                return inner
            return json.dumps(mj_expl, ensure_ascii=False)

        if isinstance(mj_expl, str) and mj_expl.strip():
            return mj_expl

    return None


def extract_q1_explanation(issue_value: Any) -> Optional[str]:
    """
    Supports:
      A) issue_value is dict with keys like "Q1", "Q2", ...
      B) issue_value is list of question objects with a 'question' field
    Returns Q1 explanation text if found.
    """
    # New shape: {"Q1": {...}, "Q2": {...}}
    if isinstance(issue_value, dict):
        if "Q1" in issue_value:
            return _extract_explanation_from_question_obj(issue_value["Q1"])

        # Fallback: try to find a question object whose "question" field mentions Question 1
        for v in issue_value.values():
            if isinstance(v, dict):
                qtxt = v.get("question", "")
                if isinstance(qtxt, str) and QUESTION_1_RE.search(qtxt):
                    return _extract_explanation_from_question_obj(v)

        return None

    # Old shape: [ {...}, {...}, ... ]
    if isinstance(issue_value, list):
        q1_obj = None

        for obj in issue_value:
            if not isinstance(obj, dict):
                continue
            qtxt = obj.get("question", "")
            if isinstance(qtxt, str) and QUESTION_1_RE.search(qtxt):
                q1_obj = obj
                break

        if q1_obj is None and issue_value and isinstance(issue_value[0], dict):
            q1_obj = issue_value[0]

        return _extract_explanation_from_question_obj(q1_obj)

    return None


def word_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(text.split())


def build_rows(multimodal: dict, text_only: dict):
    all_issues = sorted(set(multimodal.keys()) | set(text_only.keys()))

    rows = []
    for issue in all_issues:
        mm_expl = extract_q1_explanation(multimodal.get(issue))
        txt_expl = extract_q1_explanation(text_only.get(issue))

        rows.append({
            "issue": issue,  # keep exactly as in JSON (e.g., 2022Aug)
            "multimodal_word_count": word_count(mm_expl),
            "text_only_word_count": word_count(txt_expl),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--multimodal", required=True, help="Path to multimodal JSON")
    parser.add_argument("-t", "--text", required=True, help="Path to text-only JSON")
    parser.add_argument("-o", "--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    multimodal = load_json(Path(args.multimodal))
    text_only = load_json(Path(args.text))

    rows = build_rows(multimodal, text_only)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["issue", "multimodal_word_count", "text_only_word_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()


"""

python q1_exp_wc_v2.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r1.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r1.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gemini-3-pro-preview_r1_q1_word_counts.csv"
--
python q1_exp_wc_v2.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r2.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r2.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gemini-3-pro-preview_r2_q1_word_counts.csv"
--
python q1_exp_wc_v2.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r3.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r3.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gemini-3-pro-preview_r3_q1_word_counts.csv"
  

"""
