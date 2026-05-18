#!/usr/bin/env python3
# Change in V3: only one marker: filled circle, only R,G,B colos used

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from openai import OpenAI

# Initialize OpenAI client (ensure your API key is set in the environment)

# -----------------------------
# Visualization encoding
# -----------------------------

RUN_MARKERS = {k: "o" for k in [
    "r1","r2","r3","r4","r5","r6","r7","r8","r9","r10","r11","r12","r13"
]}


MODALITY_COLORS = {
    "multimodal": "red",
    "text": "blue",
    "ablation": "orange", 
}

RUN_RE = re.compile(r"(^|[-_])(r\d+)($|[-_.])", re.IGNORECASE)


# -----------------------------
# Filename inference
# -----------------------------

def detect_run(path: str) -> str:
    m = RUN_RE.search(os.path.basename(path).lower())
    if not m:
        raise ValueError(f"Cannot infer r1-r6 from filename: {path}")
    return m.group(2).lower()


def detect_modality(path: str) -> str:
    name = os.path.basename(path).lower()
    if "multimodal" in name:
        return "multimodal"
    if "image" in name:
        return "image"
    if "text" in name:
        return "text"
    if "ablation" in name:
        return "ablation"
    raise ValueError(f"Cannot infer modality from filename: {path}")


# -----------------------------
# Explanation extraction
# -----------------------------

def extract_explanation(qobj: Dict[str, Any]) -> Optional[str]:
    exp = qobj.get("explanation")
    if not exp and isinstance(qobj.get("model_json"), dict):
        exp = qobj["model_json"].get("explanation")

    if not isinstance(exp, str):
        return None

    s = exp.strip()
    if not s:
        return None

    # handle JSON-in-string (Gemini 3 style)
    if s.startswith("{") or s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                inner = parsed.get("explanation")
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
        except Exception:
            pass

    return s


def load_q1(json_path: str) -> List[Tuple[str, str]]:
    """
    Returns:
        [(issue_month, explanation_text)] for Q1 only.
    Supports:
      - month -> list: take first item
      - month -> dict: take "Q1" or first "Q*" key
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Tuple[str, str]] = []

    for issue, payload in data.items():
        q1 = None

        if isinstance(payload, list) and payload:
            q1 = payload[0]
        elif isinstance(payload, dict):
            if "Q1" in payload:
                q1 = payload["Q1"]
            else:
                keys = sorted(k for k in payload if isinstance(k, str) and k.upper().startswith("Q"))
                if keys:
                    q1 = payload[keys[0]]

        if isinstance(q1, dict):
            exp = extract_explanation(q1)
            if exp:
                rows.append((issue, exp))

    return rows


# -----------------------------
# Plotting
# -----------------------------

def plot_issue_pca(
    coords: np.ndarray,
    labels: List[str],
    modalities: List[str],
    runs: List[str],
    issue: str,
    out_dir: str,
    explained_var: Tuple[float, float],
    model_name
):
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    for i in range(len(coords)):

        # ax.text(coords[i, 0], coords[i, 1], f" {labels[i]}", fontsize=8)
        
        run = runs[i]
        color = MODALITY_COLORS[modalities[i]]

        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            marker=RUN_MARKERS[run],
            color=color,
            s=110,
            alpha=0.85,
        )


    # legends
    used_modalities = set(modalities)
    modality_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=m)
        for m, c in MODALITY_COLORS.items() if m in used_modalities
    ]
    run_handles = []
    for r in RUN_MARKERS:
        handle = plt.Line2D(
            [0], [0],
            marker=RUN_MARKERS[r],
            linestyle="",
            color="black",
            label=r,
        )
        run_handles.append(handle)


    # Move legends outside plot
    leg1 = ax.legend(
        handles=modality_handles,
        title="Modality",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        # borderaxespad=0.
    )

    leg2 = ax.legend(
        handles=run_handles,
        title="Run",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        # borderaxespad=0.
    )

    ax.add_artist(leg1)


    ax.set_title(f"{model_name}: PCA (Q1 only): MM vs Abl vs Txt : {issue}")
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
    ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pca_{issue}.png")

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(out_path, dpi=200)

    plt.close()
    print(f"Saved → {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="append", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="GPT-5.2")
    args = parser.parse_args()

    np.random.seed(args.seed)
    client = OpenAI() 

    rows: List[Tuple[str, str, str, str]] = []

    for path in args.json:
        run = detect_run(path)
        modality = detect_modality(path)

        for issue, text in load_q1(path):
            rows.append((issue, text, modality, run))

    issues = sorted(set(r[0] for r in rows))

    for issue in issues:
        subset = [r for r in rows if r[0] == issue]

        texts = [r[1] for r in subset]
        modalities = [r[2] for r in subset]
        runs = [r[3] for r in subset]

        resp = client.embeddings.create(model=args.model, input=texts)
        X = np.array([d.embedding for d in resp.data], dtype=np.float32)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        explained = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))

        plot_issue_pca(
            coords=coords,
            labels=runs,
            modalities=modalities,
            runs=runs,
            issue=issue,
            out_dir=args.outdir,
            explained_var=explained,
            model_name=args.model_name
        )


if __name__ == "__main__":
    main()


"""

Ablation 9 (r1..r10)

python mm_img_txt_pca_q1_per_issue_V3.py \
\
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r1.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r11.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r1.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r2.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r12.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r2.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r3.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r13.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r3.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r4.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r4.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r4.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r5.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r5.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r5.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r6.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r6.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r6.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r7.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r7.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r7.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r8.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r8.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r8.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r9.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r9.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r9.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-multimodal-r10.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-ablation-r10.json" \
    --json "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/gpt-5.2-2025-12-11-text-r10.json" \
\
    --outdir "/mnt/d/Naved/Outputs/jdcr_derm_vignettes/2022-2025/pca/pca_per_issue_gpt5.2-ablation-9/" \
    --model_name GPT-5.2


"""