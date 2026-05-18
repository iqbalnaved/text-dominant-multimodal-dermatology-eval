#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    Returns a list of explanation strings.
    Handles:
      - normal string explanation
      - JSON-encoded string like '[{"letter":"D","answer":"...","explanation":"..."}]'
      - JSON-encoded dict string
    """
    if not isinstance(exp, str):
        return []

    s = exp.strip()
    if not s:
        return []

    # Try parse if it looks like JSON
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


def _get_explanations(obj: Dict[str, Any]) -> List[str]:
    """
    Extract explanation(s) from a question object.
    Prefers top-level 'explanation'; falls back to obj['model_json']['explanation'].
    Also supports explanation being JSON-in-a-string.
    """
    exp = obj.get("explanation")

    # Fallback to model_json.explanation if needed
    if (not exp) and isinstance(obj.get("model_json"), dict):
        exp = obj["model_json"].get("explanation")

    # Parse or return
    exps = _extract_explanations_from_maybe_json_string(exp)
    # Filter empties
    return [e for e in (x.strip() for x in exps) if e]


def load_issue_month_texts(json_path: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of:
        (issue_month, question_label, explanation_text)

    Supports BOTH schemas:
      A) month -> list of question dicts (Gemini 2.5)
      B) month -> dict keyed by "Q1"/"Q2"/"Q3" (Gemini 3)  :contentReference[oaicite:3]{index=3}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Tuple[str, str, str]] = []

    for issue_month, month_payload in data.items():
        # Schema A: list of questions
        if isinstance(month_payload, list):
            for idx, qobj in enumerate(month_payload, start=1):
                if idx > 3:
                    continue
                if not isinstance(qobj, dict):
                    continue
                q_label = f"Q{idx}"
                for exp in _get_explanations(qobj):
                    rows.append((issue_month, q_label, exp))
            continue

        # Schema B: dict with Q1/Q2/Q3 keys
        if isinstance(month_payload, dict):
            # Prefer explicit Q1/Q2/Q3 keys if present; otherwise try any keys that start with "Q"
            qkeys = [k for k in ["Q1", "Q2", "Q3"] if k in month_payload]
            if not qkeys:
                qkeys = sorted([k for k in month_payload.keys() if isinstance(k, str) and k.upper().startswith("Q")])

            for qk in qkeys:
                qobj = month_payload.get(qk)
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


def plot_tsne(
    coords: np.ndarray,
    months: List[str],
    modalities: List[str],
    questions: List[str],
    title: str,
    out_path: str,
):
    # Colors by modality; shapes by Q#
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

    # Legend: Modality (color)
    modality_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=c, label=m)
        for m, c in modality_colors.items()
    ]
    legend1 = ax.legend(handles=modality_handles, title="Modality", loc="upper right")

    # Legend: Question (marker)
    question_handles = [
        plt.Line2D([0], [0], marker=QUESTION_MARKERS[q], linestyle="", markersize=9, color="black", label=q)
        for q in ["Q1", "Q2", "Q3"]
    ]
    legend2 = ax.legend(handles=question_handles, title="Question", loc="upper left")

    ax.add_artist(legend1)

    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot → {out_path}")


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

# needs different json parsing 
python mm_vs_txt_tsne_v2.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r1.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r1.json" \
  --title "Gemini 3 Pro (R1): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-3-pro-preview_r1.png"

python mm_vs_txt_tsne_v2.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r2.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r2.json" \
  --title "Gemini 3 Pro (R2): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-3-pro-preview_r2.png"

python mm_vs_txt_tsne_v2.py \
  --json1 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-multimodal-r3.json" \
  --json2 "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gemini-3-pro-preview-text-r3.json" \
  --title "Gemini 3 Pro (R3): t-SNE of Multimodal vs Text-only Explanations" \
  --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gemini-3-pro-preview_r3.png"
  
"""