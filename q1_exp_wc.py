#!/usr/bin/env python3

import json
import csv
import argparse
import re
from pathlib import Path
from typing import Any, Optional


QUESTION_1_RE = re.compile(r"^\s*Question\s*1\s*:", re.IGNORECASE)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_q1_explanation(issue_items: Any) -> Optional[str]:
    """
    Returns the explanation text for Question 1.
    """

    if not isinstance(issue_items, list):
        return None

    q1_obj = None

    # Look for "Question 1"
    for obj in issue_items:
        if not isinstance(obj, dict):
            continue

        question = obj.get("question", "")
        if isinstance(question, str) and QUESTION_1_RE.search(question):
            q1_obj = obj
            break

    # Fallback: assume first item is Question 1
    if q1_obj is None and issue_items:
        if isinstance(issue_items[0], dict):
            q1_obj = issue_items[0]

    if not isinstance(q1_obj, dict):
        return None

    # Explanation may appear in two places
    if isinstance(q1_obj.get("explanation"), str):
        return q1_obj["explanation"]

    model_json = q1_obj.get("model_json")
    if isinstance(model_json, dict):
        expl = model_json.get("explanation")
        if isinstance(expl, str):
            return expl

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
            "issue": issue,  # ← EXACTLY AS IN JSON
            "multimodal_word_count": word_count(mm_expl),
            "text_only_word_count": word_count(txt_expl),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--multimodal", required=True)
    parser.add_argument("-t", "--text", required=True)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    multimodal = load_json(Path(args.multimodal))
    text_only = load_json(Path(args.text))

    rows = build_rows(multimodal, text_only)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "issue",
                "multimodal_word_count",
                "text_only_word_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()


"""

python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r1.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-image-r1.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gpt5.2_r1_q1_word_counts.csv"
---
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r2.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-image-r2.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gpt5.2_r2_q1_word_counts.csv"
----

python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r3.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-image-r3.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gpt5.2_r3_q1_word_counts.csv"
----------------------------------------------------------------------------------------------------

python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r1.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r1.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gpt-o1_r1_q1_word_counts.csv"
--
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r2.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r2.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gpt-o1_r2_q1_word_counts.csv"
--
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r3.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r3.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gpt-o1_r3_q1_word_counts.csv"
  
----------------------------------------------------------------------------------------------------

python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r1.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r1.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/pixtral-large_r1_q1_word_counts.csv"
--
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r2.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r2.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/pixtral-large_r2_q1_word_counts.csv"
--
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r3.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r3.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/pixtral-large_r3_q1_word_counts.csv"


----------------------------------------------------------------------------------------------------
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r1.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r1.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gemini-3-pro-preview_r1_q1_word_counts.csv"
--
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r2.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r2.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gemini-3-pro-preview_r2_q1_word_counts.csv"
--
python q1_exp_wc.py \
  -m "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r3.json" \
  -t "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r3.json" \
  -o "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gemini-3-pro-preview_r3_q1_word_counts.csv"
  

"""