# ============================================================
# CLEAN, MODULAR, READY-TO-RUN SCRIPT
# Supports:
#   - text-only mode
#   - multimodal mode
#   - image-only mode
#   - ablation mode (vignette + wrong-case images)
#   - user-specified model via CLI (--model)
#   - record random selected images in csv (New in V3)
# ============================================================

import os
import re
import json
import base64
import random
from openai import OpenAI
import csv
from datetime import datetime

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
        m = re.match(r"^\s*([A-E])[\.\)\-:]?\s+(.*)$", line.strip(), re.IGNORECASE)
        if m:
            choices[m.group(1)] = m.group(2).strip()
    return choices


def encode_image(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")

# ============================================================
# MODEL RESPONSE PARSING
# ============================================================

def extract_text(r):
    """Extract JSON answer from OpenAI responses."""
    raw = None

    # Try output_text (o-series)
    if hasattr(r, "output_text") and r.output_text:
        raw = r.output_text

    # Try general output
    if raw is None and getattr(r, "output", None):
        for block in r.output:
            if hasattr(block, "content"):
                for c in block.content:
                    if hasattr(c, "text") and c.text:
                        raw = c.text
                        break
            if raw:
                break

    # Try summary_text
    if raw is None and hasattr(r, "summary_text"):
        raw = r.summary_text

    # Try refusal fallback
    if raw is None and hasattr(r, "refusal"):
        try:
            raw = r.refusal[0].content
        except:
            raw = None

    if not raw:
        return {"letter": "[INVALID]", "answer": "", "explanation": "No model output returned."}

    raw = raw.strip()

    # Ensure JSON parse
    try:
        return json.loads(raw)
    except:
        # fallback: try to locate JSON substring
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass
        return {"letter": "[INVALID]", "answer": raw, "explanation": "Could not parse JSON."}


def enforce_choice(model_letter, model_answer_text, choices):
    """Force model response into A–E if possible."""
    model_letter = (model_letter or "").upper().strip()
    model_answer_text = (model_answer_text or "").strip()

    # Direct match
    if model_letter in choices:
        return model_letter, choices[model_letter]

    # Match by answer text
    ans_lower = model_answer_text.lower()
    for letter, txt in choices.items():
        if ans_lower and ans_lower in txt.lower():
            return letter, txt

    # Detect stray A/B/C/D/E inside text
    m = re.search(r"\b([A-Z])\b", model_answer_text)
    if m and m.group(1) in choices:
        return m.group(1), choices[m.group(1)]

    return "[INVALID]", model_answer_text

# ============================================================
# MODEL QUERY FUNCTIONS (THREE MODES)
# ============================================================

def _base_prompt(instruction, vignette, question):
    p = (
        "You MUST respond ONLY in valid JSON.\n"
        "Return only:\n"
        '{ "letter": "<LETTER>", "answer": "<ANSWER TEXT>", "explanation": "<REASONING>" }\n'
        "No Markdown.\n\n"
        f"{instruction}\n\n"
    )
    if vignette:
        p += f"VIGNETTE:\n{vignette}\n\n"
    p += question
    return p


def ask_text(model, vignette, question):
    """TEXT-ONLY (vignette + question)."""
    p = _base_prompt("Use ONLY the vignette to answer.", vignette, question)

    r = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": p}]}]
    )
    return extract_text(r)


def ask_multimodal(model, vignette, question, encoded_images):
    """MULTIMODAL (vignette + images)."""
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
    """IMAGE-ONLY (ignore vignette)."""
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
# MAIN CASE PROCESSOR
# ============================================================

def run_cases(parent_dir, save_dir, r, eval_mode, model_name):
    """
    eval_mode = text | multimodal | image | ablation
    model_name = passed from command line
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    ablation_log_rows = []  # <-- NEW: records which images were chosen in ablation mode

    # ------------------------------------------------------------
    # Pre-index all images across cases (used by --mode ablation)
    # ------------------------------------------------------------
    case_to_image_paths = {}
    for _case_name in os.listdir(parent_dir):
        _case_dir = os.path.join(parent_dir, _case_name)
        if not os.path.isdir(_case_dir):
            continue
        paths = []
        if os.path.isdir(_case_dir):
            for f in os.listdir(_case_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(_case_dir, f))
        case_to_image_paths[_case_name] = paths

    results = {}

    # Loop over all case folders
    for case_name in os.listdir(parent_dir):
        case_dir = os.path.join(parent_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        txt_files = [f for f in os.listdir(case_dir) if f.endswith(".txt")]
        if not txt_files:
            continue

        case_file = os.path.join(case_dir, txt_files[0])
        vignette, questions = parse_case_file(case_file)

        # Load images (current case)
        image_paths = case_to_image_paths.get(case_name, [])
        encoded_images = [encode_image(p) for p in image_paths]

        # Ablation images: same count as current case, but sampled from other cases
        ablation_encoded_images = []
        if eval_mode == "ablation" and image_paths:
            pool = []
            for other_case, paths in case_to_image_paths.items():
                if other_case != case_name:
                    pool.extend(paths)

            n = len(image_paths)
            if pool:
                # Deterministic per replicate + case to make runs reproducible
                rnd = random.Random(f"{r}:{case_name}")
                if len(pool) >= n:
                    chosen = rnd.sample(pool, n)
                else:
                    chosen = [rnd.choice(pool) for _ in range(n)]
                ablation_encoded_images = [encode_image(p) for p in chosen]

                # NEW: log chosen image paths for this case + replicate
                for i, p in enumerate(chosen, start=1):
                    ablation_log_rows.append({
                        "replicate": r,
                        "case_name": case_name,
                        "chosen_index": i,
                        "chosen_image_path": p,
                        "num_images_needed": n,
                        "pool_size": len(pool),
                        "seed": f"{r}:{case_name}",
                    })

        results[case_name] = []

        # Evaluate each question
        for q in questions:
            print(f"\n===== CASE: {case_name} =====")
            print("QUESTION:")
            print(q)

            choices = extract_choices(q)

            # Select evaluation function
            def evaluate():
                if eval_mode == "text":
                    return ask_text(model_name, vignette, q)
                elif eval_mode == "multimodal":
                    return ask_multimodal(model_name, vignette, q, encoded_images)
                elif eval_mode == "image":
                    return ask_image_only(model_name, q, encoded_images)
                elif eval_mode == "ablation":
                    return ask_multimodal(model_name, vignette, q, ablation_encoded_images)
                else:
                    raise ValueError(f"Invalid eval_mode: {eval_mode}")

            raw = evaluate()
            letter, final = enforce_choice(raw["letter"], raw["answer"], choices)

            print("\n--- MODEL RESPONSE ---")
            print(f"Raw JSON: {raw}")
            print(f"Normalized Answer: {letter} - {final}")
            print(f"Explanation: {raw.get('explanation', '')}")
            print("-----------------------\n")

            results[case_name].append({
                "question": q,
                "model_json": raw,
                "answer_letter": letter,
                "answer_text": final,
                "explanation": raw.get("explanation", "")
            })

    # SAVE RESULT
    model_clean = model_name.replace("/", "_")
    outfile = os.path.join(save_dir, f"{model_clean}-{eval_mode}-r{r}.json")

    json.dump(results, open(outfile, "w"), indent=2)

    # NEW: SAVE ABLATION IMAGE SELECTIONS CSV
    if eval_mode == "ablation":
        csv_path = os.path.join(save_dir, f"{model_clean}-ablation-r{r}-selected_images.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "replicate",
                    "case_name",
                    "chosen_index",
                    "chosen_image_path",
                    "num_images_needed",
                    "pool_size",
                    "seed",
                ],
            )
            writer.writeheader()
            writer.writerows(ablation_log_rows)

        print(f"Saved ablation selected image log to: {csv_path}")

    print("\n=== DONE ===")
    print(f"Saved results to: {outfile}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("r", type=int, help="replicate number, e.g. 1,2,3")

    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["text", "multimodal", "image", "ablation"],
        help="Evaluation mode"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (e.g. gpt-5.1-2025-11-13, o1-2024-12-17)"
    )

    args = parser.parse_args()

    parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025"
    save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/ablation"

    run_cases(parent, save_dir, args.r, args.mode, args.model)


"""
python3 three_mode_eval_gpt_V3.py 4 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 5 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 6 --mode ablation --model gpt-5.2-2025-12-11

python3 three_mode_eval_gpt_V2.py 4 --mode multimodal --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 5 --mode multimodal --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 6 --mode multimodal --model gpt-5.2-2025-12-11

python3 three_mode_eval_gpt_V2.py 4 --mode text --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 5 --mode text --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 6 --mode text --model gpt-5.2-2025-12-11

python3 three_mode_eval_gpt_V2.py 4 --mode image --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 5 --mode image --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 6 --mode image --model gpt-5.2-2025-12-11

--

python3 three_mode_eval_gpt_V2.py 7 --mode multimodal --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 8 --mode multimodal --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 9 --mode multimodal --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 10 --mode multimodal --model gpt-5.2-2025-12-11

python3 three_mode_eval_gpt_V2.py 7 --mode text --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 8 --mode text --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 9 --mode text --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V2.py 10 --mode text --model gpt-5.2-2025-12-11


python3 three_mode_eval_gpt_V3.py 7 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 8 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 9 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 10 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 11 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 12 --mode ablation --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_V3.py 13 --mode ablation --model gpt-5.2-2025-12-11



"""
