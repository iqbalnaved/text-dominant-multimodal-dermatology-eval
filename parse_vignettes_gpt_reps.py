# Updated script with replicate parameter r generating four JSON files suffixed with replicate
import os
import re
import json
import base64
from openai import OpenAI

openai_api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=openai_api_key)

def parse_case_file(filepath):
    text = open(filepath).read().strip()
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

def extract_text(r):
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
            if raw is not None:
                break
    if raw is None and hasattr(r, "summary_text"):
        raw = r.summary_text
    if raw is None and hasattr(r, "refusal"):
        try:
            raw = r.refusal[0].content
        except Exception:
            raw = None
    if not raw:
        return {"letter": "[INVALID]", "answer": "", "explanation": "No model output returned."}
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "letter" in data and "answer" in data:
            if "explanation" not in data:
                data["explanation"] = ""
            return data
    except Exception:
        pass
    return {"letter": "[INVALID]", "answer": "", "explanation": raw}

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

def ask_text(model, vignette, question):
    prompt = (
        "You MUST respond ONLY in valid JSON.\n"
        "Return exactly this structure:\n"
        "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
        "No Markdown. No extra keys. No extra text.\n\n"
        "Use ONLY the vignette to answer.\n\n"
        f"VIGNETTE:\n{vignette}\n\n{question}"
    )
    r = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    )
    return extract_text(r)

def ask_multimodal(model, vignette, question, encoded_images):
    content = [{
        "type": "input_text",
        "text": (
            "You MUST respond ONLY in valid JSON.\n"
            "Return exactly this structure:\n"
            "{ \"letter\": \"<LETTER>\", \"answer\": \"<ANSWER TEXT>\", \"explanation\": \"<REASONING>\" }\n"
            "No Markdown. No extra keys. No extra text.\n\n"
            "Use BOTH the vignette AND the images to answer.\n\n"
            f"VIGNETTE:\n{vignette}\n\n{question}"
        )
    }]
    for img in encoded_images:
        content.append({"type": "input_image", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    r = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}]
    )
    return extract_text(r)

def run_cases(parent_dir, save_dir, r):
    os.makedirs(save_dir, exist_ok=True)
    o1_text = {}
    o1_multi = {}
    g51_text = {}
    g51_multi = {}
    for case_name in os.listdir(parent_dir):
        case_dir = os.path.join(parent_dir, case_name)
        if not os.path.isdir(case_dir):
            continue
        txt_files = [f for f in os.listdir(case_dir) if f.endswith(".txt")]
        if not txt_files:
            continue
        case_file = os.path.join(case_dir, txt_files[0])
        vignette, questions = parse_case_file(case_file)
        images_folder = os.path.join(case_dir, "images")
        encoded_images = []
        if os.path.isdir(images_folder):
            for img in os.listdir(images_folder):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    encoded_images.append(encode_image(os.path.join(images_folder, img)))
        o1_text[case_name] = []
        o1_multi[case_name] = []
        g51_text[case_name] = []
        g51_multi[case_name] = []
        for q in questions:
            print(f"===== CASE: {case_name} =====")
            print(f"QUESTION:{q}")

            choices = extract_choices(q)

            # o1 TEXT
            raw_json = ask_text("o1-2024-12-17", vignette, q)
            letter, final_answer = enforce_choice(raw_json.get("letter", ""), raw_json.get("answer", ""), choices)
            print("[o1 TEXT ANSWER]")
            print(f"Model JSON: {raw_json}")
            print(f"Normalized: {letter} - {final_answer}")
            print(f"Explanation: {raw_json.get('explanation','')}")
            o1_text[case_name].append({"question": q, "model_json": raw_json, "answer_letter": letter, "answer_text": final_answer, "explanation": raw_json.get("explanation", "")})

            # gpt 5.1 TEXT
            raw_json = ask_text("gpt-5.1-2025-11-13", vignette, q)
            letter, final_answer = enforce_choice(raw_json.get("letter", ""), raw_json.get("answer", ""), choices)
            print("[gpt-5.1 TEXT ANSWER]")
            print(f"Model JSON: {raw_json}")
            print(f"Normalized: {letter} - {final_answer}")
            print(f"Explanation: {raw_json.get('explanation','')}")
            g51_text[case_name].append({"question": q, "model_json": raw_json, "answer_letter": letter, "answer_text": final_answer, "explanation": raw_json.get("explanation", "")})

            # o1 MULTI
            raw_json = ask_multimodal("o1-2024-12-17", vignette, q, encoded_images)
            letter, final_answer = enforce_choice(raw_json.get("letter", ""), raw_json.get("answer", ""), choices)
            print("[o1 MULTIMODAL ANSWER]")
            print(f"Model JSON: {raw_json}")
            print(f"Normalized: {letter} - {final_answer}")
            print(f"Explanation: {raw_json.get('explanation','')}")
            o1_multi[case_name].append({"question": q, "model_json": raw_json, "answer_letter": letter, "answer_text": final_answer, "explanation": raw_json.get("explanation", "")})

            # gpt 5.1 MULTI
            raw_json = ask_multimodal("gpt-5.1-2025-11-13", vignette, q, encoded_images)
            letter, final_answer = enforce_choice(raw_json.get("letter", ""), raw_json.get("answer", ""), choices)
            print("[gpt-5.1 MULTIMODAL ANSWER]")
            print(f"Model JSON: {raw_json}")
            print(f"Normalized: {letter} - {final_answer}")
            print(f"Explanation: {raw_json.get('explanation','')}")
            g51_multi[case_name].append({"question": q, "model_json": raw_json, "answer_letter": letter, "answer_text": final_answer, "explanation": raw_json.get("explanation", "")})
            
    with open(os.path.join(save_dir, f"gpt-o1-2024-12-17-text-only-r{r}.json"), "w") as f:
        json.dump(o1_text, f, indent=2)
    with open(os.path.join(save_dir, f"gpt-o1-2024-12-17-multimodal-r{r}.json"), "w") as f:
        json.dump(o1_multi, f, indent=2)
    with open(os.path.join(save_dir, f"gpt-5.1-2025-11-13-text-only-r{r}.json"), "w") as f:
        json.dump(g51_text, f, indent=2)
    with open(os.path.join(save_dir, f"gpt-5.1-2025-11-13-multimodal-r{r}.json"), "w") as f:
        json.dump(g51_multi, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("r", type=int, help="replicate number")
    args = parser.parse_args()

    parent = "/mnt/d/Naved/Data/jaad_derm_vignette/2022-2025"
    save_dir = "/mnt/d/Naved/Outputs/jaad_derm_vignettes/2022-2025/"

    run_cases(parent, save_dir, args.r)
