import json
import os

# -------------------------
# User Option: restrict accuracy calculation to a subset of issues
# -------------------------
# Example:
# issues_to_include = ['2022Aug', '2024May', '2025Jan']

# top 10 
# issues_to_include = ['2023Dec', '2023Mar', '2022Dec', '2023Nov', '2024Mar', '2024Sep', '2023Jan', '2025Dec', '2022Oct', '2025Sep']

# bottom 10
issues_to_include = ['2022Aug', '2024May', '2025Jan', '2023May', '2022Nov', '2022Jul', '2023Apr', '2023Jun', '2023Jul', '2025Aug']   # Example: ["2022Aug", "2023Apr"]

DEBUG = False   # <--- enable debug printing

# -------------------------
# Load files
# -------------------------

gt_file = "/mnt/d/Naved/Data/jdcr_derm_vignette/cases-metadata-v0.2.json"

# output_files = [
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r1.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r2.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r3.json",
# ]

# output_files = [
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r1.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r2.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r3.json",
# ]

# output_files = [
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r1.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r2.json",
    # "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r3.json",
# ]

output_files = [
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r1.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r2.json",
    "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r3.json",
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
        year = 2000 + int(yy)
        return f"{year}{month_map[mon]}"
    except:
        return None

# -------------------------
# Process each output JSON
# -------------------------

for output_file in output_files:

    with open(output_file, "r") as f:
        gpt_data = json.load(f)

    total = 0
    correct = 0
       
    for entry in truth_data:
        issue = entry.get("Issue")
        if not issue:
            continue

        gpt_issue_key = convert_issue(issue)

        if DEBUG:
            print(f"Found issue: {issue} -> key: {gpt_issue_key}")
 
        # NEW: Skip if issue is not in the user-specified list
        if issues_to_include:
            if gpt_issue_key not in issues_to_include:
                if DEBUG:
                    print(f"   SKIP (not in include list)")
                continue
            else:
                if DEBUG:
                    print(f"   INCLUDE (in include list)")

        if not gpt_issue_key or gpt_issue_key not in gpt_data:
            print(f"   SKIP (no model output for this issue)")
            continue

        gpt_questions = gpt_data[gpt_issue_key]

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

    # Print results
    print("\n==============================")
    print(os.path.basename(output_file))
    print("==============================")

    if issues_to_include:
        print("Filtered Issues:", issues_to_include)

    print(f"Total questions evaluated: {total}")
    print(f"Correct answers: {correct}")

    if total > 0:
        print(f"Accuracy: {correct / total * 100:.2f}%\n")
    else:
        print("Accuracy: N/A (no matching issues)\n")
