# Text-Dominant Decision-Making by Large Multimodal Models in Dermatology Clinical Challenges

**Mohammad Iqbal Nouyed**, L. E. Keplinger, H. Akhter, E. Shue, D. Adjeroh, M. S. Kolodney, G. Hu

*Journal of the American Academy of Dermatology* — under review

---

## Overview

Large multimodal models (LMMs) are increasingly evaluated on clinical vignettes that pair images with text. This study reveals a fundamental failure mode: when both a clinical vignette (text) and a dermatology image are provided, models default to text-based reasoning and largely ignore the image — even when the image is diagnostically essential.

**Key finding:** Multimodal model performance on image+text input closely mirrors text-only performance, diverging significantly from image-only performance, demonstrating that models are not integrating visual evidence when text context is available.

---

## Experimental Design

Each dermatology clinical case is evaluated under three conditions:

| Mode | Input | Script |
|---|---|---|
| `text` | Vignette only (no image) | `three_mode_eval_gpt.py --mode text` |
| `multimodal` | Vignette + image | `three_mode_eval_gpt.py --mode multimodal` |
| `image` | Image only (no vignette) | `three_mode_eval_gpt.py --mode image` |

If a model truly integrates both modalities, multimodal performance should differ from both text-only and image-only. Convergence between multimodal and text-only indicates text-dominant reasoning.

---

## Repository Structure

```
text-dominant-multimodal-dermatology-eval/
│
├── Evaluation (three-mode)
│   ├── three_mode_eval_gpt.py          # GPT: text / multimodal / image modes
│   ├── three_mode_eval_gpt_V2.py       # GPT variant
│   ├── three_mode_eval_gpt_V3.py       # GPT variant
│   ├── three_mode_eval_gpt_ablation_only.py  # Ablation experiments
│   ├── three_mode_eval_gemini.py       # Gemini: three-mode eval
│   ├── three_mode_eval_mistral.py      # Mistral: three-mode eval
│   ├── batchedv2.py                    # Batched evaluation runner
│   └── one_issue_question_pair.py      # Per-issue/question evaluation
│
├── Response Generation
│   ├── generte_output_csv.py           # GPT response → CSV
│   ├── gemini_generte_output_csv.py    # Gemini response → CSV
│   ├── parse_vignettes_gpt.py          # Parse GPT vignette outputs
│   ├── parse_vignettes_gpt_reps.py     # GPT replicate parsing
│   ├── parse_vignettes_gemini_reps.py  # Gemini replicate parsing
│   ├── parse_vignettes_gemini_resume.py
│   ├── parse_vignettes_mistral_reps.py
│   └── parse_json.py                   # JSON output parser
│
├── Accuracy Analysis
│   ├── calc_accuracy.py                # Overall accuracy calculation
│   ├── calc_accuracy_by_issue_list.py  # Per-issue accuracy breakdown
│   └── extract_letter.py              # Extract MCQ answer letters
│
├── Embedding Analysis (PCA / t-SNE)
│   ├── mm_img_txt_pca_q1_per_issue.py      # PCA: image vs text embeddings per issue
│   ├── mm_img_txt_pca_q1_per_issue_V2.py
│   ├── mm_img_txt_pca_q1_per_issue_V3.py
│   ├── mm_img_txt_tsne_q1_only.py          # t-SNE: image vs text
│   ├── mm_img_txt_tsne_q1_per_issue.py     # t-SNE per issue
│   ├── mm_vs_txt_pca.py                    # Multimodal vs text-only PCA
│   ├── mm_vs_txt_tsne.py                   # Multimodal vs text-only t-SNE
│   └── mm_vs_txt_tsne_v2.py
│
├── Word Count & Vignette Analysis
│   ├── q1_exp_wc.py                    # Q1 explanation word count
│   ├── q1_exp_wc_v2.py
│   └── vignette_wc.py                  # Vignette word count analysis
│
├── Utilities
│   ├── get_default_params_gpt.py       # GPT default parameters
│   ├── get_default_params_gemini.py    # Gemini default parameters
│   ├── get_default_params_mistral.py   # Mistral default parameters
│   ├── test_gemini.py                  # Gemini API test
│   └── prompt.txt                      # Base prompt template
│
└── README.md
```

---

## Models Evaluated

| Model | Mode |
|---|---|
| GPT-5 / GPT-5.1 / GPT-5.2 / o1 (OpenAI) | text, multimodal, image |
| Gemini 1.5 Pro / Flash (Google) | text, multimodal, image |
| Mistral | text, multimodal, image |

---

## Setup

```bash
git clone https://github.com/iqbalnaved/text-dominant-multimodal-dermatology-eval.git
cd text-dominant-multimodal-dermatology-eval
pip install openai google-generativeai pandas numpy scikit-learn matplotlib seaborn
```

Set API keys:

```bash
export OPENAI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
```

---

## Usage

**Step 1 — Run three-mode evaluation**

```bash
# Text-only
python three_mode_eval_gpt.py 1 --mode text --model gpt-5.1-2025-11-13

# Multimodal (vignette + image)
python three_mode_eval_gpt.py 1 --mode multimodal --model gpt-5.1-2025-11-13

# Image-only
python three_mode_eval_gpt.py 1 --mode image --model gpt-5.1-2025-11-13
```

Each run produces a JSON output file named `{model}-{mode}-r{replicate}.json`.

**Step 2 — Calculate accuracy**

```bash
python calc_accuracy.py
python calc_accuracy_by_issue_list.py
```

**Step 3 — Embedding analysis**

```bash
# PCA: compare image vs text embedding spaces per dermatology issue
python mm_img_txt_pca_q1_per_issue.py

# t-SNE: multimodal vs text-only response clustering
python mm_vs_txt_tsne.py
```

---

## Data Format

Each case directory contains:
```
case_name/
├── case_name.txt       # Clinical vignette + MCQ questions
└── images/
    └── *.jpg           # Dermatology image(s)
```

Questions are parsed from `.txt` files. MCQ choices (A–E) are extracted automatically.

---

## Citation

> Nouyed, M. I., Keplinger, L. E., Akhter, H., Shue, E., Adjeroh, D., Kolodney, M. S., & Hu, G. (under review). Text-Dominant Decision-Making by Large Multimodal Models in Dermatology Clinical Challenges: Comment on 'AI-Assisted Dermatologic Diagnosis Using a Large Language Model.' *Journal of the American Academy of Dermatology*.

---

## Related Work

- Nouyed et al. "Sensing but Not Alerting: The Cost of Sycophancy in ChatGPT Psychodermatology Consultations." *JAAD International*, under review. [GitHub](https://github.com/iqbalnaved/sensing-not-alerting)
- Nouyed et al. "Comparative Analysis of General-Purpose vs. Domain-Specific Multimodal Models for Diabetic Retinopathy Classification." *Diagnostics*, 2026. [doi:10.3390/diagnostics16101504](https://doi.org/10.3390/diagnostics16101504)
- Akhter, Nouyed et al. "Evolving performance of GPT models in dermoscopic diagnosis." *JAAD*, 2026. [doi:10.1016/j.jaad.2025.09.119](https://doi.org/10.1016/j.jaad.2025.09.119)

---

## License

MIT License. If you use this framework, please cite the paper above.
