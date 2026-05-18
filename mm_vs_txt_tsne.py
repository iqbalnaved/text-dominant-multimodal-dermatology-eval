#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from openai import OpenAI


# Initialize OpenAI client (ensure your API key is set in the environment)
openai_api_key = os.environ['OPENAI_API_KEY']

# ----------------------------
# Configuration
# ----------------------------

QUESTION_MARKERS = {
    "Q1": "o",
    "Q2": "^",
    "Q3": "s",
}


# ----------------------------
# Helpers
# ----------------------------

def detect_modality_label(path: str) -> str:
    name = os.path.basename(path).lower()
    if "multimodal" in name:
        return "Multimodal"
    if "text" in name:
        return "Text-only"
    return os.path.splitext(os.path.basename(path))[0]


def load_issue_month_texts(json_path: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of:
        (issue_month, question_label, explanation_text)

    Example:
        ("2022Aug", "Q1", "... explanation ...")
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for issue_month, questions in data.items():
        if not isinstance(questions, list):
            continue

        for idx, obj in enumerate(questions, start=1):
            if idx > 3:
                continue

            q_label = f"Q{idx}"

            exp = obj.get("explanation")
            if not exp and isinstance(obj.get("model_json"), dict):
                exp = obj["model_json"].get("explanation")

            if isinstance(exp, str) and exp.strip():
                rows.append((issue_month, q_label, exp.strip()))

    return rows


def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    vectors = []
    batch_size = 128

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
        )
        vectors.extend([d.embedding for d in response.data])

    return np.array(vectors, dtype=np.float32)


# ----------------------------
# Plotting
# ----------------------------

def plot_tsne(
    coords: np.ndarray,
    months: List[str],
    modalities: List[str],
    questions: List[str],
    title: str,
    out_path: str,
):
    """
    Marker shape  → Q1 / Q2 / Q3
    Marker color  → modality
    """

    modality_colors = {
        "Multimodal": "#1f77b4",
        "Text-only": "#d62728",
    }

    plt.figure(figsize=(13, 10))
    ax = plt.gca()

    # Plot all points
    for i in range(len(coords)):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            marker=QUESTION_MARKERS.get(questions[i], "o"),
            s=90,
            color=modality_colors.get(modalities[i], "gray"),
            alpha=0.85,
        )

        # label month
        ax.text(
            coords[i, 0],
            coords[i, 1],
            f" {months[i]}",
            fontsize=8,
            alpha=0.85,
        )

    # -------- Legends --------

    # Question legend (marker shape)
    question_handles = [
        plt.Line2D(
            [0], [0],
            marker=QUESTION_MARKERS[q],
            linestyle="",
            markersize=9,
            color="black",
            label=q,
        )
        for q in ["Q1", "Q2", "Q3"]
    ]

    # Modality legend (color)
    modality_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=9,
            color=c,
            label=m,
        )
        for m, c in modality_colors.items()
    ]

    legend1 = ax.legend(
        handles=modality_handles,
        title="Modality",
        loc="upper right",
    )

    legend2 = ax.legend(
        handles=question_handles,
        title="Question",
        loc="upper left",
    )

    ax.add_artist(legend1)

    # -------- Formatting --------

    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot → {out_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", required=True)
    parser.add_argument("--json2", required=True)
    parser.add_argument("--out", default="tsne.png")
    parser.add_argument("--title", required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--perplexity", type=float, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()


    client = OpenAI(api_key=openai_api_key) 

    modality1 = detect_modality_label(args.json1)
    modality2 = detect_modality_label(args.json2)

    rows = []

    for month, q, text in load_issue_month_texts(args.json1):
        rows.append((month, q, text, modality1))

    for month, q, text in load_issue_month_texts(args.json2):
        rows.append((month, q, text, modality2))

    months = [r[0] for r in rows]
    questions = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    modalities = [r[3] for r in rows]

    embeddings = embed_texts(client, texts, args.model)

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )

    coords = tsne.fit_transform(embeddings)

    plot_tsne(
        coords=coords,
        months=months,
        modalities=modalities,
        questions=questions,
        title=args.title,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()


"""

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-text-r1.json" \
  --title "Gemini 2.5 Pro (R1): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-2.5-pro_r1.png"

--

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-text-r2.json" \
  --title "Gemini 2.5 Pro (R2): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-2.5-pro_r2.png"

--

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-2.5-pro-text-r3.json" \
  --title "Gemini 2.5 Pro (R3): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-2.5-pro_r3.png"


--------------------------------------------------------------------------------------------------------

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r1.json" \
  --title "GPT 5.1 (R1): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gpt-5.1_r1.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r2.json" \
  --title "GPT 5.1 (R2): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gpt-5.1_r2.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.1-2025-11-13-text-only-r3.json" \
  --title "GPT 5.1 (R3): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gpt-5.1_r3.png"
--------------------------------------------------------------------------------------------------------

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r1.json" \
  --title "GPT-o1 (R1): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gpt-o1_r1.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r2.json" \
  --title "GPT-o1 (R2): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gpt-o1_r2.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-o1-2024-12-17-text-only-r3.json" \
  --title "GPT-o1 (R3): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gpt-o1_r3.png"
--------------------------------------------------------------------------------------------------------  

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r1.json" \
  --title "Pixtral-Large (R1): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_pixtral-large_r1.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r2.json" \
  --title "Pixtral-Large (R2): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_pixtral-large_r2.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pixtral-large-2411-text-r3.json" \
  --title "Pixtral-Large (R3): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_pixtral-large_r3.png"

--------------------------------------------------------------------------------------------------------  
# needs different json parsing 
python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r1.json" \
  --title "Gemini 3 Pro (R1): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-3-pro-preview_r1.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r2.json" \
  --title "Gemini 3 Pro (R2): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-3-pro-preview_r2.png"

python mm_vs_txt_tsne.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r3.json" \
  --title "Gemini 3 Pro (R3): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-3-pro-preview_r3.png"

 
"""