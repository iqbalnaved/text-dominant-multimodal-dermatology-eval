#!/usr/bin/env python3
"""
mm_vs_txt_pca.py

Plots explanation embeddings (text-embedding-3-small) with PCA (2D).
- Supports both JSON schemas:
  A) month -> list of question dicts (Gemini 2.5 style)
  B) month -> dict with keys like "Q1","Q2","Q3" (Gemini 3 style)
- Marker shape encodes question number (Q1/Q2/Q3)
- Color encodes modality (Multimodal/Text-only) inferred from filename
- Labels each point with issue month key (e.g., "2022Aug")
- Two legends: Modality and Question

Usage example:
  python mm_vs_txt_pca.py \
    --json1 "D:/.../gemini-3-pro-preview-multimodal-r1.json" \
    --json2 "D:/.../gemini-3-pro-preview-text-r1.json" \
    --title "Gemini 3 Pro: PCA of Multimodal vs Text-only Explanations" \
    --out "D:/.../pca_gemini-3-pro.png"
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from openai import OpenAI

# Initialize OpenAI client (ensure your API key is set in the environment)
openai_api_key = os.environ['OPENAI_API_KEY']


QUESTION_MARKERS = {"Q1": "o", "Q2": "^", "Q3": "s"}


def detect_modality_label(path: str) -> str:
    name = os.path.basename(path).lower()
    if "multimodal" in name:
        return "Multimodal"
    if "text" in name:
        return "Text-only"
    return os.path.splitext(os.path.basename(path))[0]


def _extract_explanations_from_maybe_json_string(exp: Any) -> List[str]:
    """
    Gemini 3 files sometimes have 'explanation' as a JSON-encoded string.
    This returns a list of explanation strings.
    """
    if not isinstance(exp, str):
        return []
    s = exp.strip()
    if not s:
        return []

    # If it looks like JSON, try to parse
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
        except Exception:
            return [s]  # fallback: treat as plain text

        out: List[str] = []
        if isinstance(parsed, dict):
            maybe = parsed.get("explanation")
            if isinstance(maybe, str) and maybe.strip():
                out.append(maybe.strip())
            return out or [s]

        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    maybe = item.get("explanation")
                    if isinstance(maybe, str) and maybe.strip():
                        out.append(maybe.strip())
            return out or [s]

        return [s]

    return [s]


def _get_explanations(qobj: Dict[str, Any]) -> List[str]:
    """
    Extract explanation(s) from a question object.
    Prefers qobj['explanation'], falls back to qobj['model_json']['explanation'].
    Supports JSON-in-a-string explanation.
    """
    exp = qobj.get("explanation")
    if (not exp) and isinstance(qobj.get("model_json"), dict):
        exp = qobj["model_json"].get("explanation")

    exps = _extract_explanations_from_maybe_json_string(exp)
    return [e.strip() for e in exps if isinstance(e, str) and e.strip()]


def load_issue_month_texts(json_path: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (issue_month, question_label, explanation_text)
    Supports both schemas:
      A) month -> list of question dicts
      B) month -> dict keyed by Q1/Q2/Q3
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Tuple[str, str, str]] = []

    for issue_month, payload in data.items():
        # Schema A: list
        if isinstance(payload, list):
            for idx, qobj in enumerate(payload, start=1):
                if idx > 3:
                    continue
                if not isinstance(qobj, dict):
                    continue
                q_label = f"Q{idx}"
                for exp in _get_explanations(qobj):
                    rows.append((issue_month, q_label, exp))
            continue

        # Schema B: dict
        if isinstance(payload, dict):
            qkeys = [k for k in ["Q1", "Q2", "Q3"] if k in payload]
            if not qkeys:
                qkeys = sorted([k for k in payload.keys() if isinstance(k, str) and k.upper().startswith("Q")])

            for qk in qkeys:
                qobj = payload.get(qk)
                if not isinstance(qobj, dict):
                    continue
                q_label = qk.upper()
                for exp in _get_explanations(qobj):
                    rows.append((issue_month, q_label, exp))
            continue

        # Unknown schema: ignore

    return rows


def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    vectors: List[List[float]] = []
    batch_size = 128

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])

    return np.array(vectors, dtype=np.float32)


def plot_pca(
    coords: np.ndarray,
    months: List[str],
    modalities: List[str],
    questions: List[str],
    title: str,
    out_path: str,
    explained_var: Tuple[float, float],
):
    modality_colors = {
        "Multimodal": "#1f77b4",
        "Text-only": "#d62728",
    }

    plt.figure(figsize=(13, 10))
    ax = plt.gca()

    for i in range(len(coords)):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            marker=QUESTION_MARKERS.get(questions[i], "o"),
            s=90,
            color=modality_colors.get(modalities[i], "gray"),
            alpha=0.85,
        )
        ax.text(coords[i, 0], coords[i, 1], f" {months[i]}", fontsize=8, alpha=0.85)

    # Legends
    modality_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=c, label=m)
        for m, c in modality_colors.items()
    ]
    legend1 = ax.legend(handles=modality_handles, title="Modality", loc="upper right")

    question_handles = [
        plt.Line2D([0], [0], marker=QUESTION_MARKERS[q], linestyle="", markersize=9, color="black", label=q)
        for q in ["Q1", "Q2", "Q3"]
    ]
    legend2 = ax.legend(handles=question_handles, title="Question", loc="upper left")
    ax.add_artist(legend1)

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", required=True)
    parser.add_argument("--json2", required=True)
    parser.add_argument("--out", default="pca.png")
    parser.add_argument("--title", required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--seed", type=int, default=42)  # for deterministic PCA sign choices, etc.
    args = parser.parse_args()

    np.random.seed(args.seed)
    client = OpenAI(api_key=openai_api_key) 

    modality1 = detect_modality_label(args.json1)
    modality2 = detect_modality_label(args.json2)

    rows: List[Tuple[str, str, str, str]] = []
    for month, q, text in load_issue_month_texts(args.json1):
        rows.append((month, q, text, modality1))
    for month, q, text in load_issue_month_texts(args.json2):
        rows.append((month, q, text, modality2))

    if not rows:
        raise RuntimeError("No explanations found in either JSON file.")

    months = [r[0] for r in rows]
    questions = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    modalities = [r[3] for r in rows]

    embeddings = embed_texts(client, texts, args.model)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    explained = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))

    plot_pca(
        coords=coords,
        months=months,
        modalities=modalities,
        questions=questions,
        title=args.title,
        out_path=args.out,
        explained_var=explained,
    )


if __name__ == "__main__":
    main()

"""

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-text-r1.json" \
  --title "Gemini 2.5 Pro (R1): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gemini-2.5-pro_r1.png"

--

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-text-r2.json" \
  --title "Gemini 2.5 Pro (R2): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gemini-2.5-pro_r2.png"

--

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-text-r3.json" \
  --title "Gemini 2.5 Pro (R3): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gemini-2.5-pro_r3.png"


--------------------------------------------------------------------------------------------------------

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r1.json" \
  --title "GPT 5.1 (R1): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gpt-5.1_r1.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r2.json" \
  --title "GPT 5.1 (R2): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gpt-5.1_r2.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r3.json" \
  --title "GPT 5.1 (R3): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gpt-5.1_r3.png"
--------------------------------------------------------------------------------------------------------

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r1.json" \
  --title "GPT-o1 (R1): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gpt-o1_r1.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r2.json" \
  --title "GPT-o1 (R2): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gpt-o1_r2.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r3.json" \
  --title "GPT-o1 (R3): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gpt-o1_r3.png"
--------------------------------------------------------------------------------------------------------  

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r1.json" \
  --title "Pixtral-Large (R1): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_pixtral-large_r1.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r2.json" \
  --title "Pixtral-Large (R2): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_pixtral-large_r2.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r3.json" \
  --title "Pixtral-Large (R3): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_pixtral-large_r3.png"

--------------------------------------------------------------------------------------------------------  
# needs different json parsing 
python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r1.json" \
  --title "Gemini 3 Pro (R1): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gemini-3-pro-preview_r1.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r2.json" \
  --title "Gemini 3 Pro (R2): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gemini-3-pro-preview_r2.png"

python mm_vs_txt_pca.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r3.json" \
  --title "Gemini 3 Pro (R3): PCA of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_gemini-3-pro-preview_r3.png"

  
"""