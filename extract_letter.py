import json
import re
import csv
from datetime import datetime

# -------- INPUT FILE ----------
input_file = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/ablation/gpt-5.2-2025-12-11-text-r6.json"

# -------- OUTPUT FILE ----------
output_file = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/csv/gpt-5.2-2025-12-11-text-r6.csv"

# ---------- YOUR EXACT ORDER ----------
order_text = """
Aug-24	Q3
Sep-24	Q2
Oct-24	Q3
Nov-24	Q3
Dec-24	Q1
Jan-25	Q2
Apr-25	Q3
Jul-25	Q1
Jun-22	Q1
Jun-22	Q2
Jul-22	Q1
Oct-22	Q2
Dec-22	Q2
Dec-22	Q3
Feb-23	Q1
Apr-23	Q3
May-23	Q3
Aug-23	Q3
Sep-23	Q2
Nov-23	Q3
Dec-23	Q2
Dec-25	Q2
Feb-24	Q3
Mar-24	Q2
May-24	Q1
May-24	Q3
Jun-24	Q2
Jun-24	Q3
Jul-24	Q2
Sep-24	Q1
Nov-24	Q1
Jan-25	Q1
Jan-25	Q3
Feb-25	Q1
Mar-25	Q1
May-25	Q1
May-25	Q3
Aug-25	Q1
May-22	Q1
Aug-22	Q3
Sep-22	Q1
Sep-22	Q3
Jan-23	Q3
Mar-23	Q2
Jun-23	Q1
Jun-23	Q2
Sep-23	Q1
Oct-23	Q2
Dec-23	Q3
Apr-24	Q2
Jun-24	Q1
Jul-24	Q1
Jul-24	Q3
Oct-24	Q1
Feb-25	Q2
Mar-25	Q3
Apr-25	Q1
May-25	Q2
Sep-25	Q2
Oct-25	Q1
Nov-25	Q2
May-22	Q3
Aug-22	Q1
Dec-22	Q1
Jan-23	Q2
Feb-23	Q2
Apr-23	Q1
May-23	Q1
Jun-23	Q3
Aug-23	Q1
Oct-23	Q3
Nov-23	Q1
Dec-25	Q1
Jan-24	Q1
Jan-24	Q2
Feb-24	Q1
Mar-24	Q1
Apr-24	Q1
Sep-24	Q3
Nov-24	Q2
Dec-24	Q2
Mar-25	Q2
Apr-25	Q2
Jun-25	Q1
Sep-25	Q1
Nov-25	Q1
May-22	Q2
Jul-22	Q3
Aug-22	Q2
Sep-22	Q2
Oct-22	Q3
Nov-22	Q1
Nov-22	Q2
Nov-22	Q3
Jan-23	Q1
Mar-23	Q1
Mar-23	Q3
Jul-23	Q1
Jul-23	Q3
Aug-23	Q2
Nov-23	Q2
Dec-23	Q1
Jan-24	Q3
Feb-24	Q2
Mar-24	Q3
Apr-24	Q3
May-24	Q2
Aug-24	Q1
Aug-24	Q2
Oct-24	Q2
Dec-24	Q3
Feb-25	Q3
Jun-25	Q2
Jun-25	Q3
Nov-25	Q3
Jun-22	Q3
Jul-22	Q2
Oct-22	Q1
Feb-23	Q3
Apr-23	Q2
May-23	Q2
Jul-23	Q2
Sep-23	Q3
Oct-23	Q1
"""

# Convert order text to list
desired_order = [tuple(line.split()) for line in order_text.strip().split("\n")]

order_lookup = {v: i for i, v in enumerate(desired_order)}

# ---------- FORMAT ISSUE ----------
def format_issue(issue_key):
    match = re.match(r"(\d{4})([A-Za-z]{3})", issue_key)
    if match:
        year = match.group(1)[-2:]
        month = match.group(2)
        return f"{month}-{year}"
    return issue_key

# ---------- LOAD JSON ----------
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
found_keys = set()

# ---------- EXTRACT LETTERS ----------
for issue, questions in data.items():
    formatted_issue = format_issue(issue)

    for idx, q in enumerate(questions, start=1):
        q_number = f"Q{idx}"
        letter = q.get("answer_letter")

        if letter:
            key = (formatted_issue, q_number)
            rows.append((formatted_issue, q_number, letter))
            found_keys.add(key)

# ---------- SORT USING YOUR ORDER ----------
rows.sort(key=lambda x: order_lookup.get((x[0], x[1]), 999999))

# ---------- WARN IF ORDER ITEM MISSING ----------
missing = [k for k in desired_order if k not in found_keys]
if missing:
    print("⚠ Missing entries in JSON:")
    for m in missing:
        print(m)

# ---------- WRITE CSV ----------
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Issue","Question","Letter"])
    writer.writerows(rows)

print("✅ CSV saved")
