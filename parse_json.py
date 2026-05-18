import json
import csv
import re

src = "/mnt/d/Naved/Outputs/jaad_derm_vignettes/2023/gpt-o1-2024-12-17-text-only.json"
dest = "/mnt/d/Naved/Outputs/jaad_derm_vignettes/2023/gpt-o1-2024-12-17-text-only.csv"

# ---- Load your JSON file ----
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Prepare CSV output ----
with open(dest, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["yearmonth", "question#", "letter", "answer"])

    # Iterate through each year-month block
    for yearmonth, questions in data.items():
        for entry in questions:

            # Extract question number from text (assumes "Question X:")
            match = re.search(r"Question\s*(\d+)", entry["question"])
            qnum = match.group(1) if match else ""

            # Extract letter and answer text
            letter = entry.get("answer_letter", "")
            answer = entry.get("answer_text", "")

            writer.writerow([yearmonth, qnum, letter, answer])

print("CSV created successfully: questions_output.csv")
