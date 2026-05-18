# ===============================================================
# Patched Script — Gemini Version (Batched)
# ===============================================================

import os
import re
import json
import base64
import google.generativeai as genai


# ===============================================================
# Parsing / Utility functions
# ===============================================================

def parse_case_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    parts = re.split(r"(Question\s*\d+:)", text)
    if len(parts) < 3:
        raise ValueError(f"No questions found in {filepath}")
    vignette = parts[0].strip()
    questions = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip()
        questions.append(f"{header}\n{body}")
    return vignette, questions


def extract_choices(question_text):
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
# Gemini JSON extraction
# ===============================================================

def extract_text_gemini(response):
    """
    Gemini returns response.text or structured candidates.
    This wraps and extracts raw JSON text.
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
        return {
            "letter": "[INVALID]",
            "answer": "",
            "explanation": "No model output returned."
        }

    try:
        data = json.loads(raw)
        return data
    except Exception:
        return raw


# ===============================================================
# NEW: Batched Gemini text-only call
# ===============================================================

def ask_text_batch_gemini(model_name, vignette, questions):
    model = genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = (
        "Respond ONLY in valid JSON.\n"
        "Return a list where each item is:\n"
        "{ \"question_index\": <INDEX>, \"letter\": \"<LETTER>\", "
        "\"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n\n"
        f"VIGNETTE:\n{vignette}\n\n"
        f"QUESTIONS:\n{json.dumps(questions)}"
    )

    # Correct API usage:
    response = model.generate_content([{"text": prompt}])

    print("RAW RESPONSE:", response)

    raw = extract_text_gemini(response)

    if not isinstance(raw, list):
        raw = [{
            "question_index": i+1,
            "letter": "[INVALID]",
            "answer": "",
            "explanation": f"Invalid format: {raw}"
        } for i in range(len(questions))]

    return raw



# ===============================================================
# NEW: Batched Gemini multimodal call
# ===============================================================

def ask_multimodal_batch_gemini(model_name, vignette, questions, encoded_images):
    model = genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json"}
    )

    text_block = (
        "Respond ONLY in valid JSON.\n"
        "Return a list where each item is:\n"
        "{ \"question_index\": <INDEX>, \"letter\": \"<LETTER>\", "
        "\"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n\n"
        f"VIGNETTE:\n{vignette}\n\n"
        f"QUESTIONS:\n{json.dumps(questions)}"
    )

    parts = [{"text": text_block}]

    for img_b64 in encoded_images:
        parts.append({"image": base64.b64decode(img_b64)})

    response = model.generate_content(parts)
    
    print("RAW RESPONSE:", response)
    
    raw = extract_text_gemini(response)

    if not isinstance(raw, list):
        raw = [{
            "question_index": i+1,
            "letter": "[INVALID]",
            "answer": "",
            "explanation": f"Invalid multimodal format: {raw}"
        } for i in range(len(questions))]

    return raw



# ===============================================================
# Normalize model answer → enforce valid choice
# ===============================================================

def enforce_choice(model_letter, model_answer_text, choices):
    model_letter = (model_letter or "").strip().upper()
    model_answer_text = (model_answer_text or "").strip()

    if not choices:
        return model_letter or "[INVALID]", model_answer_text

    if model_letter in choices:
        return model_letter, choices[model_letter]

    ans_lower = model_answer_text.lower()
    for letter, text in choices.items():
        if ans_lower and ans_lower in text.lower():
            return letter, text

    m = re.search(r"\b([A-Z])\b", model_answer_text)
    if m:
        cand = m.group(1)
        if cand in choices:
            return cand, choices[cand]

    return "[INVALID]", model_answer_text


# ===============================================================
# NEW: Main case runner (batched version)
# ===============================================================

def run_cases(parent_dir, save_dir, r, model_name):
    os.makedirs(save_dir, exist_ok=True)

    text_results = {}
    multi_results = {}

    for case_name in os.listdir(parent_dir):
        case_dir = os.path.join(parent_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        txt_files = [f for f in os.listdir(case_dir) if f.endswith(".txt")]
        if not txt_files:
            continue

        case_file = os.path.join(case_dir, txt_files[0])
        vignette, questions = parse_case_file(case_file)

        # Load images
        images_folder = os.path.join(case_dir, "images")
        encoded_images = []
        if os.path.isdir(images_folder):
            for img in os.listdir(images_folder):
                if img.lower().endswith((".jpg", ".jpeg")):
                    encoded_images.append(encode_image(os.path.join(images_folder, img)))

        print(f"\n===== CASE: {case_name} =====")
        print(f"Questions: {len(questions)}")

        # ---------------------------------------------------------------
        # BATCHED TEXT-ONLY CALL
        # ---------------------------------------------------------------
        text_batch = ask_text_batch_gemini(model_name, vignette, questions)

        text_results[case_name] = []
        for i, item in enumerate(text_batch):
            q = questions[i]
            choices = extract_choices(q)

            letter, final_answer = enforce_choice(
                item.get("letter", ""),
                item.get("answer", ""),
                choices
            )

            text_results[case_name].append({
                "question": q,
                "model_json": item,
                "answer_letter": letter,
                "answer_text": final_answer,
                "explanation": item.get("explanation", "")
            })

        # ---------------------------------------------------------------
        # BATCHED MULTIMODAL CALL
        # ---------------------------------------------------------------
        multi_batch = ask_multimodal_batch_gemini(model_name, vignette, questions, encoded_images)

        multi_results[case_name] = []
        for i, item in enumerate(multi_batch):
            q = questions[i]
            choices = extract_choices(q)

            letter, final_answer = enforce_choice(
                item.get("letter", ""),
                item.get("answer", ""),
                choices
            )

            multi_results[case_name].append({
                "question": q,
                "model_json": item,
                "answer_letter": letter,
                "answer_text": final_answer,
                "explanation": item.get("explanation", "")
            })


    # ===============================================================
    # Save outputs (1 text + 1 multimodal per replicate)
    # ===============================================================
    with open(os.path.join(save_dir, f"{model_name}-text-r{r}.json"), "w") as f:
        json.dump(text_results, f, indent=2)

    with open(os.path.join(save_dir, f"{model_name}-multimodal-r{r}.json"), "w") as f:
        json.dump(multi_results, f, indent=2)


# ===============================================================
# CLI Entry
# ===============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", type=int, required=True, help="replicate number")
    parser.add_argument("-k", type=int, required=True, help="api-key number")
    parser.add_argument("--model", required=True, help="Gemini model name")

    args = parser.parse_args()

    # ===============================================================
    # Gemini Setup
    # ===============================================================
    keychain = [
        os.envion['GEMINI_API_KEY1'],
        os.envion['GEMINI_API_KEY2'],
        os.envion['GEMINI_API_KEY3']
    ]

    GEMINI_API_KEY = keychain[args.k]
    genai.configure(api_key=GEMINI_API_KEY)

    parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025"
    save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/"

    run_cases(parent, save_dir, args.r, args.model)



"""


python3 parse_vignettes_gemini_batched.py -r 1 -k 0 --model gemini-3-pro-preview
python3 parse_vignettes_gemini_batched.py -r 2 -k 1 --model gemini-3-pro-preview
python3 parse_vignettes_gemini_batched.py -r 3 -k 2 --model gemini-3-pro-preview

"""