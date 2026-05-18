# ============================================================
# CLEAN, MODULAR, READY-TO-RUN SCRIPT
# Supports:
#   - text-only mode
#   - multimodal mode
#   - image-only mode
#   - ablation mode (vignette + wrong-case images)
#   - user-specified model via CLI (--model)
#   - record random selected images in csv
#   - OPTIONAL single case + single question execution
# ============================================================

import os
import re
import json
import base64
import random
from openai import OpenAI
import csv

# ============================================================
# CONFIG
# ============================================================

client = OpenAI()

# ============================================================
# FILE PARSING
# ============================================================

def parse_case_file(filepath):
    """Extract vignette + individual questions from text file."""
    text = open(filepath).read().strip()
    parts = re.split(r"(Question\s*\d+:)", text)

    if len(parts) < 3:
        raise ValueError(f"No questions found in {filepath}")

    vignette = parts[0].strip()
    questions = []

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip()
        questions.append(f"{header}\n{body}")

    return vignette, questions


def extract_choices(question_text):
    """Extract MCQ choices A–E from the question."""
    choices = {}
    for line in question_text.splitlines():
        m = re.match(r"^\s*([A-E])[\.\)\-]?\s+(.*)$", line.strip(), re.IGNORECASE)
        if m:
            choices[m.group(1).upper()] = m.group(2).strip()
    return choices


def encode_image(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")

# ============================================================
# MODEL RESPONSE PARSING
# ============================================================

def extract_text(r):
    """Extract JSON answer from OpenAI responses."""
    raw = None

    if hasattr(r, "output_text") and r.output_text:
        raw = r.output_text

    if raw is None and getattr(r, "output", None):
        for block in r.output:
            if hasattr(block, "content"):
                for c in block.content:
                    if hasattr(c, "text") and c.text:
                        raw = c.text
                        break
            if raw:
                break

    if raw is None and hasattr(r, "summary_text"):
        raw = r.summary_text

    if not raw:
        return {
            "letter": "[INVALID]",
            "answer": "",
            "explanation": "No model output returned."
        }

    raw = raw.strip()

    try:
        return json.loads(raw)
    except:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass

    return {
        "letter": "[INVALID]",
        "answer": raw,
        "explanation": "Could not parse JSON."
    }


def enforce_choice(model_letter, model_answer_text, choices):
    model_letter = (model_letter or "").upper().strip()
    model_answer_text = (model_answer_text or "").strip()

    if model_letter in choices:
        return model_letter, choices[model_letter]

    ans_lower = model_answer_text.lower()
    for letter, txt in choices.items():
        if ans_lower and ans_lower in txt.lower():
            return letter, txt

    m = re.search(r"\b([A-E])\b", model_answer_text.upper())
    if m and m.group(1) in choices:
        return m.group(1), choices[m.group(1)]

    return "[INVALID]", model_answer_text

# ============================================================
# PROMPTS
# ============================================================

def _base_prompt(instruction, vignette, question):
    p = (
        "You MUST respond ONLY in valid JSON.\n"
        '{ "letter": "<LETTER>", "answer": "<ANSWER TEXT>", "explanation": "<REASONING>" }\n'
        "No Markdown.\n\n"
        f"{instruction}\n\n"
    )
    if vignette:
        p += f"VIGNETTE:\n{vignette}\n\n"
    p += question
    return p


def ask_text(model, vignette, question):
    p = _base_prompt("Use ONLY the vignette to answer.", vignette, question)
    r = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": p}]}]
    )
    return extract_text(r)


def ask_multimodal(model, vignette, question, encoded_images):
    content = [{
        "type": "input_text",
        "text": _base_prompt("Use BOTH the vignette AND the images.", vignette, question)
    }]
    for img in encoded_images:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{img}"
        })

    r = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}]
    )
    return extract_text(r)


def ask_image_only(model, question, encoded_images):
    content = [{
        "type": "input_text",
        "text": _base_prompt("Use ONLY the images. Ignore the vignette.", None, question)
    }]
    for img in encoded_images:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{img}"
        })

    r = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}]
    )
    return extract_text(r)

# ============================================================
# MAIN RUNNER
# ============================================================

def run_cases(parent_dir, r, eval_mode, model_name,
              case_filter=None, question_filter=None):

    # os.makedirs(save_dir, exist_ok=True)
    ablation_log_rows = {}

    # Pre-index images
    case_to_image_paths = {}
    for cname in os.listdir(parent_dir):
        cdir = os.path.join(parent_dir, cname)
        if not os.path.isdir(cdir):
            continue
        case_to_image_paths[cname] = [
            os.path.join(cdir, f)
            for f in os.listdir(cdir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    results = {}

    for case_name in os.listdir(parent_dir):
        if case_filter and case_name != case_filter:
            continue

        case_dir = os.path.join(parent_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        txt_files = [f for f in os.listdir(case_dir) if f.endswith(".txt")]
        if not txt_files:
            continue

        vignette, questions = parse_case_file(
            os.path.join(case_dir, txt_files[0])
        )

        if question_filter is not None:
            idx = question_filter - 1
            if idx < 0 or idx >= len(questions):
                raise ValueError(f"{case_name} has no Question {question_filter}")
            questions = [questions[idx]]

        image_paths = case_to_image_paths.get(case_name, [])
        encoded_images = [encode_image(p) for p in image_paths]

        ablation_images = []
        if eval_mode == "ablation" and image_paths:
            pool = [
                p for k, v in case_to_image_paths.items()
                if k != case_name for p in v
            ]
            rnd = random.Random(f"{r}:{case_name}")
            chosen = rnd.sample(pool, len(image_paths))
            ablation_images = [encode_image(p) for p in chosen]
            ablation_log_rows[case_name] = chosen

        results[case_name] = []

        for q in questions:
            choices = extract_choices(q)

            if eval_mode == "text":
                raw = ask_text(model_name, vignette, q)
            elif eval_mode == "multimodal":
                raw = ask_multimodal(model_name, vignette, q, encoded_images)
            elif eval_mode == "image":
                raw = ask_image_only(model_name, q, encoded_images)
            elif eval_mode == "ablation":
                raw = ask_multimodal(model_name, vignette, q, ablation_images)
            else:
                raise ValueError("Invalid mode")

            letter, final = enforce_choice(
                raw.get("letter"),
                raw.get("answer"),
                choices
            )

            results[case_name].append({
                "question": q,
                "model_json": raw,
                "answer_letter": letter,
                "answer_text": final,
                "explanation": raw.get("explanation", "")
            })

    model_clean = model_name.replace("/", "_")
    
    print(results)
    # out = os.path.join(
        # save_dir,
        # f"{model_clean}-{eval_mode}-r{r}.json"
    # )
    # json.dump(results, open(out, "w"), indent=2)
    # print(f"Saved results to {out}")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("r", type=int, help="replicate number")

    parser.add_argument(
        "--mode",
        default="multimodal",
        choices=["text", "multimodal", "image", "ablation"]
    )

    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run only one case folder (e.g. 2024May)"
    )

    parser.add_argument(
        "--question",
        type=int,
        default=None,
        help="Run only one question number (e.g. 1)"
    )

    args = parser.parse_args()

    parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025"
    # save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/ablation"

    run_cases(
        parent,
        # save_dir,
        args.r,
        args.mode,
        args.model,
        case_filter=args.case,
        question_filter=args.question,
    )


"""

python3 one_issue_question_pair.py \
    4 \
    --mode image \
    --model gpt-5.2-2025-12-11 \
    --case 2024May \
    --question 1

"""