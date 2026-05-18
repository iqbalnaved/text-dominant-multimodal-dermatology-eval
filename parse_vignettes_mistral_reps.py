# ===============================================================
# Patched Script — Mistral Version
# ===============================================================

import os
import re
import json
import base64
from mistralai import Mistral


# ===============================================================
# Parsing / Utility functions (unchanged)
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
# Mistral JSON extraction
# ===============================================================

def extract_text_mistral(response):
    """
    Mistral returns response.choices[0].message.content
    """
    try:
        raw = response.choices[0].message.content.strip()
    except Exception:
        return {
            "letter": "[INVALID]",
            "answer": "",
            "explanation": "No model output returned."
        }

    # Try to parse JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "letter" in data and "answer" in data:
            data.setdefault("explanation", "")
            return data
    except Exception:
        pass

    return {
        "letter": "[INVALID]",
        "answer": "",
        "explanation": raw
    }


# ===============================================================
# Mistral text-only query
# ===============================================================

def ask_text_mistral(client, model_name, vignette, question):
    prompt = (
        "You MUST respond ONLY in valid JSON.\n"
        "Return exactly this structure:\n"
        "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
        "No Markdown. No extra keys. No extra text.\n\n"
        "Use ONLY the vignette to answer.\n\n"
        f"VIGNETTE:\n{vignette}\n\n{question}"
    )

    response = client.chat.complete(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_text_mistral(response)


# ===============================================================
# Mistral multimodal query (images supported via base64)
# ===============================================================

def ask_multimodal_mistral(client, model_name, vignette, question, encoded_images):
    parts = [{
        "type": "text",
        "text": (
            "You MUST respond ONLY in valid JSON.\n"
            "Return exactly this structure:\n"
            "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
            "No Markdown. No extra keys. No extra text.\n\n"
            "Use BOTH the vignette AND the images to answer.\n\n"
            f"VIGNETTE:\n{vignette}\n\n{question}"
        )
    }]

    for img_b64 in encoded_images:
        parts.append({
            "type": "image",
            "image": img_b64,
            "mime_type": "image/jpeg"
        })

    response = client.chat.complete(
        model=model_name,
        messages=[{"role": "user", "content": parts}]
    )
    return extract_text_mistral(response)


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
# Main case runner
# ===============================================================

def run_cases(parent_dir, save_dir, r, model_name, api_key):
    os.makedirs(save_dir, exist_ok=True)

    client = Mistral(api_key=api_key)

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

        # Load images (base64 list)
        images_folder = os.path.join(case_dir, "images")
        encoded_images = []
        if os.path.isdir(images_folder):
            for img in os.listdir(images_folder):
                if img.lower().endswith((".jpg", ".jpeg")):
                    encoded_images.append(encode_image(os.path.join(images_folder, img)))

        text_results[case_name] = []
        multi_results[case_name] = []

        # -------------------------------------------------------
        # Loop questions
        # -------------------------------------------------------
        for q in questions:
            print(f"===== CASE: {case_name} =====")
            print(f"QUESTION: {q}")

            choices = extract_choices(q)

            # TEXT ONLY
            raw_json = ask_text_mistral(client, model_name, vignette, q)
            letter, final_answer = enforce_choice(raw_json.get("letter", ""), raw_json.get("answer", ""), choices)
            print(f"{model_name} TEXT")
            print("Model JSON:", raw_json)
            print("Normalized:", letter, "-", final_answer)
            text_results[case_name].append({
                "question": q,
                "model_json": raw_json,
                "answer_letter": letter,
                "answer_text": final_answer,
                "explanation": raw_json.get("explanation", "")
            })

            # MULTIMODAL
            raw_json = ask_multimodal_mistral(client, model_name, vignette, q, encoded_images)
            letter, final_answer = enforce_choice(raw_json.get("letter", ""), raw_json.get("answer", ""), choices)
            print(f"{model_name} MULTIMODAL")
            print("Model JSON:", raw_json)
            print("Normalized:", letter, "-", final_answer)
            multi_results[case_name].append({
                "question": q,
                "model_json": raw_json,
                "answer_letter": letter,
                "answer_text": final_answer,
                "explanation": raw_json.get("explanation", "")
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
    parser.add_argument("--model", required=True, help="Mistral model name (e.g., mistral-medium)")

    args = parser.parse_args()

    keychain = [
        "hg36lIQZ2iMstCVUORJoxPt0r32AbiaF",
        "58OHd8kUdpLcpxy1PwYezUnC40JrGHx5",
        "KB2ssO8jj6x2HOD4SEHRmdVXYTxRXSDP"
    ]

    api_key = keychain[args.k]

    parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025"
    save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/"

    run_cases(parent, save_dir, args.r, args.model, api_key)

"""

python3 parse_vignettes_mistral_reps.py -r 1 -k 0 --model pixtral-large-2411

python3 parse_vignettes_mistral_reps.py -r 2 -k 1 --model pixtral-large-2411

python3 parse_vignettes_mistral_reps.py -r 3 -k 2 --model pixtral-large-2411

"""
