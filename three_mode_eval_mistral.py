# ===============================================================
# Updated Script — Adds image-only mode, evaluation selector,
# vignette skipping, and prints model answers
# ===============================================================

import os
import re
import json
import base64
from mistralai import Mistral


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
# JSON extractor
# ===============================================================

def extract_text_mistral(response):
    try:
        raw = response.choices[0].message.content.strip()
    except Exception:
        return {
            "letter": "[INVALID]",
            "answer": "",
            "explanation": "No model output returned."
        }

    # NEW: Remove markdown code fences (```json ... ```)
    raw = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

    # Try to parse JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "letter" in data and "answer" in data:
            data.setdefault("explanation", "")
            return data
    except Exception:
        pass

    # Otherwise return raw text
    return {
        "letter": "[INVALID]",
        "answer": "",
        "explanation": raw
    }



# ===============================================================
# Query Functions
# ===============================================================

def ask_text_mistral(client, model_name, vignette, question):
    prompt = (
        "You MUST respond ONLY in valid JSON.\n"
        "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
        "Use ONLY the vignette to answer.\n\n"
        f"VIGNETTE:\n{vignette}\n\n{question}"
    )
    response = client.chat.complete(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_text_mistral(response)


def ask_multimodal_mistral(client, model_name, vignette, question, encoded_images):
    prompt = (
        "You MUST respond ONLY in valid JSON.\n"
        "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
        "Use BOTH the vignette AND the images.\n\n"
        f"VIGNETTE:\n{vignette}\n\n{question}"
    )

    response = client.chat.complete(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "image_urls": [f"data:image/jpeg;base64,{img}" for img in encoded_images]
            }
        ]
    )
    return extract_text_mistral(response)



def ask_image_only_mistral(client, model_name, question, encoded_images):
    prompt = (
        "You MUST respond ONLY in valid JSON.\n"
        "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
        "Use ONLY the images to answer. IGNORE ALL TEXT.\n\n"
        f"{question}"
    )

    response = client.chat.complete(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "image_urls": [f"data:image/jpeg;base64,{img}" for img in encoded_images]
            }
        ]
    )
    return extract_text_mistral(response)



# ===============================================================
# Normalize model answer
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

def run_cases(parent_dir, save_dir, r, model_name, api_key, eval_mode):
    os.makedirs(save_dir, exist_ok=True)
    client = Mistral(api_key=api_key)

    results = {}

    for case_name in os.listdir(parent_dir):
        case_dir = os.path.join(parent_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        txt_files = [f for f in os.listdir(case_dir) if f.endswith(".txt")]
        if not txt_files:
            continue

        case_file = os.path.join(case_dir, txt_files[0])
        vignette, questions = parse_case_file(case_file)

        # Load images from the same directory as case.txt
        encoded_images = []
        for img in os.listdir(case_dir):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(case_dir, img)
                encoded_images.append(encode_image(full_path))

        print("IMAGES FOUND:", len(encoded_images))


        results[case_name] = []

        for q in questions:
            print("\n==============================")
            print(f"CASE: {case_name}")
            print("QUESTION:")
            print(q)
            print("Evaluation mode:", eval_mode)

            choices = extract_choices(q)

            # -----------------------------
            # HANDLE MODES
            # -----------------------------
            if eval_mode == "text":
                raw_json = ask_text_mistral(client, model_name, vignette, q)

            elif eval_mode == "multimodal":
                raw_json = ask_multimodal_mistral(client, model_name, vignette, q, encoded_images)

            elif eval_mode == "image":
                raw_json = ask_image_only_mistral(client, model_name, q, encoded_images)

            else:
                raise ValueError("Invalid eval mode")

            # Normalize
            letter, final_answer = enforce_choice(
                raw_json.get("letter", ""),
                raw_json.get("answer", ""),
                choices
            )

            print("MODEL OUTPUT:", raw_json)
            print("FINAL ANSWER:", letter, "-", final_answer)

            results[case_name].append({
                "question": q,
                "model_json": raw_json,
                "answer_letter": letter,
                "answer_text": final_answer,
                "explanation": raw_json.get("explanation", "")
            })

    # Save file
    out_file = os.path.join(save_dir, f"{model_name}-{eval_mode}-r{r}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out_file)


# ===============================================================
# CLI Entry
# ===============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", type=int, required=True, help="replicate number")
    parser.add_argument("-k", type=int, required=True, help="api-key number")
    parser.add_argument("--model", required=True, help="model name")
    parser.add_argument("--mode", required=True, choices=["text", "multimodal", "image"],
                        help="Choose evaluation mode")

    args = parser.parse_args()

    keychain = [
        os.environ['MISTRAL_API_KEY1'],
        os.environ['MISTRAL_API_KEY2'],
        os.environ['MISTRAL_API_KEY3']
    ]

    api_key = keychain[args.k]

    # parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025"
    # save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/"

    parent = "/mnt/d/Naved/Data/jdcr_derm_vignette/incomplete"
    save_dir = "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/incomplete"

    run_cases(parent, save_dir, args.r, args.model, api_key, args.mode)


"""
python3 three_mode_eval_mistral.py -r 1 -k 0 --mode multimodal --model pixtral-large-2411
python3 three_mode_eval_mistral.py -r 2 -k 1 --mode multimodal --model pixtral-large-2411
python3 three_mode_eval_mistral.py -r 3 -k 2 --mode multimodal --model pixtral-large-2411

python3 three_mode_eval_mistral.py -r 1 -k 0 --mode text --model pixtral-large-2411
python3 three_mode_eval_mistral.py -r 2 -k 1 --mode text --model pixtral-large-2411
python3 three_mode_eval_mistral.py -r 3 -k 2 --mode text --model pixtral-large-2411

python3 three_mode_eval_mistral.py -r 1 -k 0 --mode image --model pixtral-large-2411
python3 three_mode_eval_mistral.py -r 2 -k 1 --mode image --model pixtral-large-2411
python3 three_mode_eval_mistral.py -r 3 -k 2 --mode image --model pixtral-large-2411


"""