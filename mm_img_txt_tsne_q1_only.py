#!/usr/bin/env python3
"""
mm_img_txt_tsne_runs.py

t-SNE over explanation embeddings for:
- modalities: multimodal (red), image-only (green), text-only (blue)
- runs: r1 (circle), r2 (square), r3 (triangle)
- ONLY FIRST QUESTION (Q1) per issue month.

Supports both JSON schemas:
A) month -> list of question dicts (Gemini 2.5 style)
B) month -> dict keyed by "Q1"/"Q2"/"Q3" (Gemini 3 style), plus cases where
   explanation is JSON-in-a-string.

USAGE example:
  python mm_img_txt_tsne_runs.py \
    --json "D:/.../gemini-3-pro-preview-multimodal-r1.json" \
    --json "D:/.../gemini-3-pro-preview-image-r1.json" \
    --json "D:/.../gemini-3-pro-preview-text-r1.json" \
    --json "D:/.../gemini-3-pro-preview-multimodal-r2.json" \
    --json "D:/.../gemini-3-pro-preview-image-r2.json" \
    --json "D:/.../gemini-3-pro-preview-text-r2.json" \
    --json "D:/.../gemini-3-pro-preview-multimodal-r3.json" \
    --json "D:/.../gemini-3-pro-preview-image-r3.json" \
    --json "D:/.../gemini-3-pro-preview-text-r3.json" \
    --title "Gemini 3 Pro: t-SNE (Q1 only) - Modality color, Run marker" \
    --out "D:/.../tsne_runs.png"
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from openai import OpenAI

# Initialize OpenAI client (ensure your API key is set in the environment)
openai_api_key = os.environ['OPENAI_API_KEY']

# Run -> marker
RUN_MARKERS = {
    "r1": "o",  # circle
    "r2": "s",  # square
    "r3": "^",  # triangle
}

# Modality -> color
MODALITY_COLORS = {
    "multimodal": "red",
    "image": "green",
    "text": "blue",
}

RUN_RE = re.compile(r"(^|[-_])(r[123])($|[-_.])", re.IGNORECASE)


def detect_run(path: str) -> str:
    """Infer r1/r2/r3 from filename."""
    name = os.path.basename(path).lower()
    m = RUN_RE.search(name)
    if not m:
        raise ValueError(f"Could not infer run (r1/r2/r3) from filename: {path}")
    return m.group(2).lower()


def detect_modality(path: str) -> str:
    """
    Infer modality from filename.
    Expects one of: 'multimodal', 'image'/'image-only', 'text'/'text-only'
    """
    name = os.path.basename(path).lower()
    if "multimodal" in name:
        return "multimodal"
    # be generous for image-only naming
    if "image-only" in name or "imageonly" in name or re.search(r"(^|[-_])image($|[-_.])", name):
        return "image"
    if "text-only" in name or "textonly" in name or "text" in name:
        return "text"
    raise ValueError(f"Could not infer modality (multimodal/image/text) from filename: {path}")


def _extract_explanations_from_maybe_json_string(exp: Any) -> List[str]:
    """
    Handles:
      - normal string explanation
      - JSON-encoded string like '[{"letter":"D","answer":"...","explanation":"..."}]'
      - JSON-encoded dict string
    Returns list of explanation strings.
    """
    if not isinstance(exp, str):
        return []

    s = exp.strip()
    if not s:
        return []

    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
        except Exception:
            return [s]  # treat as plain text if parsing fails

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


def _get_explanation_text(qobj: Dict[str, Any]) -> Optional[str]:
    """
    Extract a single explanation string from a question object.
    Prefers top-level 'explanation'; falls back to qobj['model_json']['explanation'].
    If explanation is JSON-in-a-string, returns the first parsed explanation.
    """
    exp = qobj.get("explanation")
    if (not exp) and isinstance(qobj.get("model_json"), dict):
        exp = qobj["model_json"].get("explanation")

    exps = _extract_explanations_from_maybe_json_string(exp)
    exps = [e.strip() for e in exps if isinstance(e, str) and e.strip()]
    return exps[0] if exps else None


def load_q1_explanations(json_path: str) -> List[Tuple[str, str]]:
    """
    Returns list of (issue_month, explanation_text) for Q1 only.
    Supports:
      A) month -> list (take first element)
      B) month -> dict (take 'Q1' if present; else first Q* key)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Tuple[str, str]] = []

    for issue_month, payload in data.items():
        q1_obj: Optional[Dict[str, Any]] = None

        # Schema A: list of questions -> first is Q1
        if isinstance(payload, list) and payload:
            if isinstance(payload[0], dict):
                q1_obj = payload[0]

        # Schema B: dict keyed by Q1/Q2/Q3
        elif isinstance(payload, dict) and payload:
            if isinstance(payload.get("Q1"), dict):
                q1_obj = payload["Q1"]
            else:
                # fallback: first key that starts with Q (sorted for stability)
                qkeys = sorted([k for k in payload.keys() if isinstance(k, str) and k.upper().startswith("Q")])
                if qkeys and isinstance(payload.get(qkeys[0]), dict):
                    q1_obj = payload[qkeys[0]]

        if q1_obj:
            exp = _get_explanation_text(q1_obj)
            if exp:
                rows.append((issue_month, exp))

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
    runs: List[str],
    title: str,
    out_path: str,
):
    plt.figure(figsize=(13, 10))
    ax = plt.gca()

    # Plot points
    for i in range(len(coords)):
        modality = modalities[i]
        run = runs[i]
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            marker=RUN_MARKERS.get(run, "o"),
            s=90,
            color=MODALITY_COLORS.get(modality, "gray"),
            alpha=0.85,
        )
        ax.text(coords[i, 0], coords[i, 1], f" {months[i]}", fontsize=8, alpha=0.85)

    # Legend 1: Modality (color)
    modality_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=MODALITY_COLORS[m], label=m)
        for m in ["multimodal", "image", "text"]
        if m in MODALITY_COLORS
    ]
    legend1 = ax.legend(handles=modality_handles, title="Modality (color)", loc="upper right")

    # Legend 2: Run (marker)
    run_handles = [
        plt.Line2D([0], [0], marker=RUN_MARKERS[r], linestyle="", markersize=9, color="black", label=r)
        for r in ["r1", "r2", "r3"]
        if r in RUN_MARKERS
    ]
    legend2 = ax.legend(handles=run_handles, title="Run (marker)", loc="upper left")
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
    parser.add_argument(
        "--json",
        action="append",
        required=True,
        help="Path to a JSON file. Provide multiple times (expects r1/r2/r3 across modalities).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--perplexity", type=float, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client = OpenAI(api_key=openai_api_key) 
    
    # Build dataset: one point per (issue_month, file)
    months: List[str] = []
    texts: List[str] = []
    modalities: List[str] = []
    runs: List[str] = []

    for path in args.json:
        run = detect_run(path)
        modality = detect_modality(path)

        for issue_month, exp in load_q1_explanations(path):
            months.append(issue_month)
            texts.append(exp)
            modalities.append(modality)
            runs.append(run)

    if not texts:
        raise RuntimeError("No Q1 explanations found across provided JSON files.")

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
        runs=runs,
        title=args.title,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()


"""

python mm_img_txt_tsne_q1_only.py \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r1.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-image-r1.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r1.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r2.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-image-r2.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r2.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r3.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-image-r3.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r3.json" \
    --title "GPT-5.2: t-SNE (Q1 only) - MM vs Img vs Txt - Triplicates" \
    --out "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/tsne/tsne_gpt-5.2_q1_runs.png"
"""