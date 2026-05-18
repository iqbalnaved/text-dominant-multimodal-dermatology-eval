import json
import os

# -------------------------
# Load files
# -------------------------

gt_file = "/mnt/d/Naved/Data/jdcr_derm_vignette/cases-metadata-v0.2.json"

# output_files = [
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-image-r1.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-image-r2.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-image-r3.json",
# ]

output_files = [
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r1.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r11.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r1.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r2.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r12.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r2.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r3.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r13.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r3.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r4.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r4.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r4.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r5.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r5.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r5.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r6.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r6.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r6.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r7.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r7.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r7.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r8.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r8.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r8.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r9.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r9.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r9.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r10.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r10.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r10.json"
]

with open(gt_file, "r") as f:
    truth_data = json.load(f)

# -------------------------
# Helper: Convert "Apr-23" -> "2023Apr"
# -------------------------
def convert_issue(issue_str):
    month_map = {
        "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr",
        "May": "May", "Jun": "Jun", "Jul": "Jul", "Aug": "Aug",
        "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dec": "Dec"
    }
    try:
        mon, yy = issue_str.split("-")
        year = 2000 + int(yy)   # e.g. "23" → 2023
        return f"{year}{month_map[mon]}"
    except:
        return None

# -------------------------
# Process each output JSON
# -------------------------

for output_file in output_files:

    # Load model output
    with open(output_file, "r") as f:
        gpt_data = json.load(f)

    total = 0
    correct = 0

    for entry in truth_data:
        issue = entry.get("Issue")
        if not issue:
            continue

        gpt_issue_key = convert_issue(issue)

        if not gpt_issue_key or gpt_issue_key not in gpt_data:
            continue

        gpt_questions = gpt_data[gpt_issue_key]

        # comment out for gemini 3
        for q_num in [1, 2, 3]:
            truth_answer = entry.get(f"Q{q_num}")
            if truth_answer is None:
                continue

            if len(gpt_questions) < q_num:
                continue

            gpt_answer = gpt_questions[q_num - 1].get("model_json", {}).get("letter")
            if not gpt_answer:
                continue

            total += 1
            if gpt_answer == truth_answer:
                correct += 1

        # uncomment for gemini-3-pro-preview
        # for q_num in [1, 2, 3]:
            # truth_answer = entry.get(f"Q{q_num}")
            # if truth_answer is None:
                # continue

            # gpt_q_data = gpt_questions.get(f"Q{q_num}", {})
            # gpt_answer = gpt_q_data.get("model_json", {}).get("letter")

            # if not gpt_answer:
                # continue

            # total += 1
            # if gpt_answer == truth_answer:
                # correct += 1
        

    # Print results for this file
    print("\n==============================")
    print(os.path.basename(output_file))
    print("==============================")
    print(f"Total questions evaluated: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%\n")
