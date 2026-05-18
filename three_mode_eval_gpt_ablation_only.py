# ============================================================
# CLEAN, MODULAR, READY-TO-RUN SCRIPT
# Supports:
#   - text-only mode
#   - multimodal mode
#   - image-only mode
#   - user-specified model via CLI (--model)
# ============================================================

import os
import re
import json
import base64
import random
from openai import OpenAI

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
            choices[m.group(1)] = m.group(2).strip()
    return choices


def encode_image(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")


def list_image_paths(img_dir):
    """Return sorted list of image file paths in a directory."""
    if not os.path.isdir(img_dir):
        return []
    paths = []
    for f in os.listdir(img_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(img_dir, f))
    return sorted(paths)

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

    # Try JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "letter" in parsed and "answer" in parsed:
            parsed.setdefault("explanation", "")
            return parsed
    except:
        pass

    # If JSON invalid → Return explanation only
    return {
        "letter": "[INVALID]",
        "answer": "",
        "explanation": raw
    }


def enforce_choice(model_letter, model_answer_text, choices):
    """Normalize model output to a valid choice A–E if possible."""
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
        "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
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

def run_cases(parent_dir, save_dir, r, eval_mode, model_name, ablate_images=False):
    """
    eval_mode = text | multimodal | image
    model_name = passed from command line
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # ------------------------------------------------------------
    # Pre-index *all* images across cases once (used for ablation).
    # We sort everything for deterministic behavior given a seed.
    # ------------------------------------------------------------
    rng = random.Random(r)  # replicate number controls sampling deterministically
    all_case_names = sorted([d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])

    images_by_case = {}
    all_image_paths = []
    for cname in all_case_names:
        cdir = os.path.join(parent_dir, cname)
        # c_img_dir = os.path.join(cdir, "images")
        c_paths = list_image_paths(cdir)
        images_by_case[cname] = c_paths
        all_image_paths.extend(c_paths)

    all_image_paths = sorted(set(all_image_paths))

    # Loop over all case folders
    for case_name in all_case_names:
        case_dir = os.path.join(parent_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        txt_files = [f for f in os.listdir(case_dir) if f.endswith(".txt")]
        if not txt_files:
            continue

        case_file = os.path.join(case_dir, txt_files[0])
        vignette, questions = parse_case_file(case_file)


        # Load images for this case
        case_img_paths = images_by_case.get(case_name, [])
        n_case_imgs = len(case_img_paths)

        # ------------------------------------------------------------
        # Image ablation:
        #   - keep the *count* of images the same for the case
        #   - replace them with random images sampled from the global pool
        #     excluding this case's own images
        # ------------------------------------------------------------
        if ablate_images and eval_mode in {"multimodal", "image"} and n_case_imgs > 0:
            excluded = set(case_img_paths)
            candidate_paths = [p for p in all_image_paths if p not in excluded]

            if not candidate_paths:
                raise RuntimeError(
                    f"Ablation requested but no candidate images exist outside case '{case_name}'."
                )

            # Sample without replacement when possible; otherwise sample with replacement.
            if len(candidate_paths) >= n_case_imgs:
                sampled_paths = rng.sample(candidate_paths, n_case_imgs)
            else:
                sampled_paths = [rng.choice(candidate_paths) for _ in range(n_case_imgs)]

            img_paths_to_use = sampled_paths
        else:
            img_paths_to_use = case_img_paths

        encoded_images = [encode_image(p) for p in img_paths_to_use]

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
    outfile = os.path.join(save_dir, f"{model_clean}-{eval_mode}-r{r}-ablation.json")

    json.dump(results, open(outfile, "w"), indent=2)

    print("\n=== DONE ===")
    print(f"Saved results to: {outfile}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("r", type=int, help="replicate number, e.g., 1,2,3")

    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["text", "multimodal", "image"],
        help="Evaluation mode"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (e.g., gpt-5.1-2025-11-13, o1-2024-12-17)"
    )

    parser.add_argument(
        "--ablate_images",
        action="store_true",
        help="If set (and mode is multimodal/image), replace each case's images with an equal number of randomly sampled images from other cases."
    )

    args = parser.parse_args()

    parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025/"
    save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/"

    run_cases(parent, save_dir, args.r, args.mode, args.model, ablate_images=args.ablate_images)


"""
python3 three_mode_eval_gpt_ablation.py 1 --mode multimodal --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_ablation.py 2 --mode multimodal --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_ablation.py 3 --mode multimodal --model gpt-5.2-2025-12-11

---

python3 three_mode_eval_gpt_ablation.py 1 --mode text --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_ablation.py 2 --mode text --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_ablation.py 3 --mode text --model gpt-5.2-2025-12-11

---

python3 three_mode_eval_gpt_ablation.py 1 --mode image --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_ablation.py 2 --mode image --model gpt-5.2-2025-12-11
python3 three_mode_eval_gpt_ablation.py 3 --mode image --model gpt-5.2-2025-12-11

"""