# ===============================================================
# Patched Script — Gemini Version
# Now supports:
#   ✔ Resume per question
#   ✔ Automatic retry with exponential backoff
#   ✔ Incremental saving after every question
#   ✔ Mode selection: text / multimodal / image-only
#   ✔ Image-only mode ignores vignette, uses images + question only
# ===============================================================

import os
import re
import json
import base64
import time
import random
import google.generativeai as genai


# ===============================================================
# Parsing / Utility functions
# ===============================================================

def parse_case_file(filepath):
    """
    Parse a case file into:
      - vignette (string before first 'Question <n>:')
      - list of full question blocks (including 'Question n:' header and choices)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()

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
    """
    Extract multiple-choice options from question text.
    Returns dict mapping letter -> text, e.g. {'A': 'foo', 'B': 'bar', ...}
    """
    choices = {}
    lines = question_text.splitlines()
    for line in lines:
        m = re.match(r"([A-Z])\.\s*(.*)", line.strip())
        if m:
            letter = m.group(1)
            text = m.group(2).strip()
            choices[letter] = text
    return choices


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ===============================================================
# Exponential Backoff Wrapper
# ===============================================================

def retry_with_backoff(func, max_retries=5, base_delay=2):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[Retry {attempt+1}/{max_retries}] Error: {e}. Waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries.")


# ===============================================================
# Gemini JSON extraction
# ===============================================================

def extract_text_gemini(response):
    """
    Extract JSON from a Gemini response.
    Expected JSON format:
      { "letter": "<LETTER>", "answer": "<ANSWER TEXT>", "explanation": "<REASONING>" }
    """
    raw = ""

    if response and getattr(response, "text", None):
        raw = response.text.strip()

    if not raw and hasattr(response, "candidates"):
        try:
            raw = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            pass

    if not raw:
        return {"letter": "[INVALID]", "answer": "", "explanation": "No model output returned."}

    try:
        data = json.loads(raw)
        data.setdefault("explanation", "")
        return data
    except Exception:
        # Not valid JSON; store raw string in explanation
        return {"letter": "[INVALID]", "answer": "", "explanation": raw}


# ===============================================================
# Gemini Queries (Text / Multimodal / Image-only)
# ===============================================================

def ask_text_gemini(model_name, vignette, question):
    """
    Text-only: use the vignette + question, no images.
    """
    model = genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = (
        "You MUST respond ONLY in valid JSON.\n"
        "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
        "Use ONLY the vignette and question text below.\n\n"
        f"VIGNETTE:\n{vignette}\n\n{question}"
    )

    r = model.generate_content(prompt)
    return extract_text_gemini(r)


def ask_multimodal_gemini(model_name, vignette, question, encoded_images):
    """
    Multimodal: use the vignette + question + images.
    """
    model = genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json"}
    )

    parts = [{
        "text": (
            "You MUST respond ONLY in valid JSON.\n"
            "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
            "Use BOTH the vignette AND the image(s) along with the question.\n\n"
            f"VIGNETTE:\n{vignette}\n\n{question}"
        )
    }]

    for img_b64 in encoded_images:
        parts.append({
            "mime_type": "image/jpeg",
            "data": base64.b64decode(img_b64),
        })

    r = model.generate_content(parts)
    return extract_text_gemini(r)


def ask_image_only_gemini(model_name, question, encoded_images):
    """
    Image-only mode: IGNORE the vignette/clinical text.
    Use ONLY the image(s) PLUS the question text and answer choices.
    """
    model = genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json"}
    )

    parts = [{
        "text": (
            "You MUST respond ONLY in valid JSON.\n"
            "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
            "You are answering a multiple-choice question using the IMAGE(S) and the QUESTION TEXT ONLY.\n"
            "Completely IGNORE any clinical vignette or case description that might exist elsewhere.\n\n"
            "Question and choices:\n"
            f"{question}"
        )
    }]

    for img_b64 in encoded_images:
        parts.append({
            "mime_type": "image/jpeg",
            "data": base64.b64decode(img_b64),
        })

    r = model.generate_content(parts)
    return extract_text_gemini(r)


# ===============================================================
# Normalize model answer → valid choice
# ===============================================================

def enforce_choice(model_letter, model_answer_text, choices):
    """
    Given model's letter + free-text answer and a dict of choices,
    try to map to a valid multiple-choice option.
    """
    model_letter = (model_letter or "").strip().upper()
    model_answer_text = (model_answer_text or "").strip()

    if not choices:
        return model_letter or "[INVALID]", model_answer_text

    # If letter is valid, done
    if model_letter in choices:
        return model_letter, choices[model_letter]

    # Try to match by substring of answer text
    ans_lower = model_answer_text.lower()
    for letter, text in choices.items():
        if ans_lower and ans_lower in text.lower():
            return letter, text

    # Try to extract lone capital letter from answer text
    m = re.search(r"\b([A-Z])\b", model_answer_text)
    if m and m.group(1) in choices:
        return m.group(1), choices[m.group(1)]

    # Fallback: invalid letter, keep text as-is
    return "[INVALID]", model_answer_text


# ===============================================================
# Main runner - Resume Per Question + Incremental Save + Modes
# ===============================================================

def run_cases(parent_dir, save_dir, r, model_name, mode):
    """
    Run all cases under parent_dir using:
      mode = 'text'  → text-only (vignette + question)
      mode = 'multimodal' → multimodal (vignette + question + images)
      mode = 'image' → image-only (images + question, vignette ignored)

    Results are stored in:
      {save_dir}/{model_name}-{mode}-r{r}.json

    with structure:
      {
        "CASE_NAME": {
          "Q1": { ... },
          "Q2": { ... },
          ...
        },
        ...
      }
    """
    os.makedirs(save_dir, exist_ok=True)

    outfile = os.path.join(save_dir, f"{model_name}-{mode}-r{r}.json")
    results = json.load(open(outfile)) if os.path.exists(outfile) else {}

    # ==========================================================
    # Loop through cases
    # ==========================================================
    for case_name in sorted(os.listdir(parent_dir)):
        case_dir = os.path.join(parent_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        txt_files = [f for f in os.listdir(case_dir) if f.endswith(".txt")]
        if not txt_files:
            continue

        print(f"\n===== STARTING CASE: {case_name} =====")

        case_file = os.path.join(case_dir, txt_files[0])
        vignette, questions = parse_case_file(case_file)

        # Ensure case dict exists
        results.setdefault(case_name, {})

        # Load images
        images_folder = os.path.join(case_dir, "images")
        encoded_images = []
        if os.path.isdir(images_folder):
            for img in sorted(os.listdir(images_folder)):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    encoded_images.append(encode_image(os.path.join(images_folder, img)))

        # ======================================================
        # Loop questions — RESUME PER QUESTION
        # ======================================================
        for q_idx, q in enumerate(questions):
            q_key = f"Q{q_idx+1}"

            if q_key in results[case_name]:
                print(f"--- Skipping {case_name} {q_key} (already saved for mode={mode}) ---")
                continue

            print(f"\n--- Running {case_name} {q_key} [mode={mode}] ---")

            # Print full question text, including choices
            print("\nQUESTION TEXT:\n" + "-"*60)
            print(q)
            print("-"*60 + "\n")

            choices = extract_choices(q)


            # Call appropriate evaluation function
            if mode == "text":
                raw = retry_with_backoff(lambda: ask_text_gemini(model_name, vignette, q))

            elif mode == "multimodal":
                raw = retry_with_backoff(lambda: ask_multimodal_gemini(
                    model_name, vignette, q, encoded_images
                ))

            elif mode == "image":
                raw = retry_with_backoff(lambda: ask_image_only_gemini(
                    model_name, q, encoded_images
                ))

            else:
                raise ValueError(f"Invalid mode: {mode}")

            # Normalize to a valid choice
            letter, final_answer = enforce_choice(
                raw.get("letter", ""), raw.get("answer", ""), choices
            )

            # Store result
            results[case_name][q_key] = {
                "question": q,
                "model_json": raw,
                "answer_letter": letter,
                "answer_text": final_answer,
                "explanation": raw.get("explanation", "")
            }

            # Incremental save
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)

            # Print model answer
            print(f"Model {model_name} [mode={mode}] → {letter}: {final_answer}")
            print(f"Saved {case_name} {q_key}")

        print(f"===== COMPLETED CASE: {case_name} =====")


# ===============================================================
# CLI Entry
# ===============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=int, required=True, help="replicate number")
    parser.add_argument("-k", type=int, required=True, help="api-key index")
    parser.add_argument("--model", required=True, help="Gemini model name")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["text", "multimodal", "image"],
        help=(
            "Evaluation mode: "
            "'text'  → vignette + question (no images); "
            "'multimodal' → vignette + question + images; "
            "'image' → images + question only (vignette ignored)"
        ),
    )
    args = parser.parse_args()

    # Gemini Setup (using your existing keychain)
    keychain = [
        os.environ['GEMINI_API_KEY1'],
        os.environ['GEMINI_API_KEY2'],
        os.environ['GEMINI_API_KEY3']"
    ]

    gemini_api_key = keychain[args.k]
    genai.configure(api_key=gemini_api_key)

    # parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025"
    # save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/"

    parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/incomplete"
    save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/incomplete"
    
    run_cases(parent, save_dir, args.r, args.model, args.mode)


"""
Example usage:

# Text-only (vignette + question)
python3 parse_vignettes_gemini_resume.py -r 1 -k 0 --model gemini-3-pro-preview --mode text

# Multimodal (vignette + question + images)
python3 parse_vignettes_gemini_resume.py -r 1 -k 0 --model gemini-3-pro-preview --mode multi

# Image-only (images + question, vignette ignored)
python3 parse_vignettes_gemini_resume.py -r 1 -k 0 --model gemini-3-pro-preview --mode image
"""




"""

python3 three_mode_eval_gemini.py -r 1 -k 0 --model gemini-3-pro-preview --mode multimodal
python3 three_mode_eval_gemini.py -r 2 -k 1 --model gemini-3-pro-preview --mode multimodal
python3 three_mode_eval_gemini.py -r 3 -k 2 --model gemini-3-pro-preview --mode multimodal

python3 three_mode_eval_gemini.py -r 1 -k 0 --model gemini-3-pro-preview --mode text
python3 three_mode_eval_gemini.py -r 2 -k 1 --model gemini-3-pro-preview --mode text
python3 three_mode_eval_gemini.py -r 3 -k 2 --model gemini-3-pro-preview --mode text

python3 three_mode_eval_gemini.py -r 1 -k 0 --model gemini-3-pro-preview --mode image
python3 three_mode_eval_gemini.py -r 2 -k 1 --model gemini-3-pro-preview --mode image
python3 three_mode_eval_gemini.py -r 3 -k 2 --model gemini-3-pro-preview --mode image


"""
