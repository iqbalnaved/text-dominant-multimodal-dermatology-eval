# Text-Dominant Decision-Making by Large Multimodal Models in Dermatology Clinical Challenges

**Mohammad Iqbal Nouyed**, Lauren E. Kozlowski, Halima Akhter, Evelyn Shue, Donald A. Adjeroh, Michael S. Kolodney, Gangqing Hu

*Journal of the American Academy of Dermatology* (2026) — [doi:10.1016/j.jaad.2026.05.129](https://doi.org/10.1016/j.jaad.2026.05.129)

> **Article type:** Letter: Notes & Comments. Comment on *"AI-assisted dermatologic diagnosis using a large language model"* (Dezoteux et al., *JAAD*, 2025).

---

## Overview

Large multimodal models (LMMs) are increasingly evaluated on clinical vignettes that pair images with text. This study reveals a fundamental failure mode: when both a clinical vignette (text) and a dermatology image are provided, models default to text-based reasoning and largely ignore the image — even when the image is diagnostically essential.

**Key findings:**
- Multimodal (text-plus-images) accuracy was statistically indistinguishable from text-only accuracy (*t*-test *P* > 0.05) across all four models tested.
- Images-only accuracy dropped to 43–58%, significantly below multimodal/text-only performance (*P* < 0.05), but remained above random guessing (20%).
- Answer choices were consistent across runs for >80% of questions under text-plus-images and text-only conditions.
- Replacing the correct image with an irrelevant image left ChatGPT-5.2 accuracy virtually unchanged (87.4% vs 87.6%).
- Over 80% of image-anchored explanatory instances merely paraphrased the vignette stem text rather than contributing image-derived reasoning.

---

## Results

**Table I. Performance of LMMs on JDCR Case Challenges** (mean ± SD, 3 runs per model)

| Condition | ChatGPT-o1 | ChatGPT-5.2 | Gemini 3 Pro | Pixtral |
|---|---|---|---|---|
| Text-plus-images | 81.5% ± 2.2% | 87.6% ± 1.7% | 86.0% ± 0.5% | 78.5% ± 1.7% |
| Text-only | 82.0% ± 1.3% | 89.2% ± 1.2% | 85.8% ± 1.2% | 77.7% ± 2.0% |
| Images-only | 48.9% ± 1.2% | 50.5% ± 2.6% | 57.8% ± 4.7% | 43.0% ± 2.9% |

Individual runs deviated from their mean by < 3.0%.

---

## Dataset

**JDCR Case Challenge** — 44 eligible cases (124 questions), published May 2022 – January 2026. Each case includes a textual vignette, 2–3 dermatology images, and 1–3 multiple-choice questions (5 options). Cases published before May 2022 were excluded due to absent answers in the main text.

**Supplementary material** (Tables 1–6, Figure 1): [https://data.mendeley.com/datasets/tmd5kd2m2h/3](https://data.mendeley.com/datasets/tmd5kd2m2h/3)

---

## Experimental Design

Each case was evaluated under three conditions using identical questions and answer options:

| Mode | Input | Script |
|---|---|---|
| `text` | Vignette only (no image) | `three_mode_eval_gpt.py --mode text` |
| `multimodal` | Vignette + image | `three_mode_eval_gpt.py --mode multimodal` |
| `image` | Image only (no vignette) | `three_mode_eval_gpt.py --mode image` |

If a model truly integrates both modalities, multimodal performance should differ from both text-only and image-only. Convergence between multimodal and text-only is the signature of text-dominant reasoning.

---

## Models Evaluated

| Model | API string | Provider |
|---|---|---|
| GPT-o1 | `o1-2024-12-17` | OpenAI |
| ChatGPT-5.2 | `gpt-5.2-2025-12-11` | OpenAI |
| Gemini 3 Pro | `gemini-3-pro-preview` | Google |
| Pixtral | `pixtral-large-2411` | Mistral AI |

All models evaluated under default settings. Each experiment run 3 times per model.

---

## Repository Structure

```
text-dominant-multimodal-dermatology-eval/
│
├── Evaluation (three-mode)
│   ├── three_mode_eval_gpt.py               # GPT: text / multimodal / image modes
│   ├── three_mode_eval_gpt_V2.py
│   ├── three_mode_eval_gpt_V3.py
│   ├── three_mode_eval_gpt_ablation_only.py # Ablation (irrelevant image replacement)
│   ├── three_mode_eval_gemini.py            # Gemini: three-mode eval
│   ├── three_mode_eval_mistral.py           # Pixtral: three-mode eval
│   ├── batchedv2.py                         # Batched evaluation runner
│   └── one_issue_question_pair.py           # Per-case/question evaluation
│
├── Response Generation
│   ├── generte_output_csv.py                # GPT response → CSV
│   ├── gemini_generte_output_csv.py         # Gemini response → CSV
│   ├── parse_vignettes_gpt.py
│   ├── parse_vignettes_gpt_reps.py
│   ├── parse_vignettes_gemini_reps.py
│   ├── parse_vignettes_gemini_resume.py
│   ├── parse_vignettes_mistral_reps.py
│   └── parse_json.py
│
├── Accuracy Analysis
│   ├── calc_accuracy.py                     # Overall accuracy calculation
│   ├── calc_accuracy_by_issue_list.py       # Per-case accuracy breakdown
│   └── extract_letter.py                    # Extract MCQ answer letters
│
├── Embedding Analysis (PCA / t-SNE)
│   ├── mm_img_txt_pca_q1_per_issue.py
│   ├── mm_img_txt_pca_q1_per_issue_V2.py
│   ├── mm_img_txt_pca_q1_per_issue_V3.py
│   ├── mm_img_txt_tsne_q1_only.py
│   ├── mm_img_txt_tsne_q1_per_issue.py
│   ├── mm_vs_txt_pca.py                     # Multimodal vs text-only PCA
│   ├── mm_vs_txt_tsne.py
│   └── mm_vs_txt_tsne_v2.py
│
├── Word Count & Vignette Analysis
│   ├── q1_exp_wc.py
│   ├── q1_exp_wc_v2.py
│   └── vignette_wc.py
│
├── Utilities
│   ├── get_default_params_gpt.py
│   ├── get_default_params_gemini.py
│   ├── get_default_params_mistral.py
│   ├── test_gemini.py
│   └── prompt.txt
│
└── README.md
```

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
export MISTRAL_API_KEY=your_key_here
```

---

## Usage

**Step 1 — Run three-mode evaluation**

```bash
# Text-only
python three_mode_eval_gpt.py 1 --mode text --model gpt-5.2-2025-12-11

# Multimodal (vignette + image)
python three_mode_eval_gpt.py 1 --mode multimodal --model gpt-5.2-2025-12-11

# Image-only
python three_mode_eval_gpt.py 1 --mode image --model gpt-5.2-2025-12-11
```

Each run produces a JSON output file named `{model}-{mode}-r{replicate}.json`.

**Step 2 — Calculate accuracy**

```bash
python calc_accuracy.py
python calc_accuracy_by_issue_list.py
```

**Step 3 — Embedding analysis**

```bash
# PCA: compare image vs text embedding spaces per dermatology case
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

```bibtex
@article{nouyed2026textdominant,
  title   = {Text-dominant decision-making by large multimodal models in dermatology
             clinical challenges. Comment on ``{AI}-assisted dermatologic diagnosis
             using a large language model''},
  author  = {Iqbal Nouyed, Mohammad and Kozlowski, Lauren E. and Akhter, Halima and
             Shue, Evelyn and Adjeroh, Donald A. and Kolodney, Michael S. and Hu, Gangqing},
  journal = {Journal of the American Academy of Dermatology},
  year    = {2026},
  doi     = {10.1016/j.jaad.2026.05.129}
}
```

---

## Related Work

- Nouyed et al. "Sensing but Not Alerting: The Cost of Sycophancy in ChatGPT Psychodermatology Consultations." *JAAD International*, under review. [GitHub](https://github.com/iqbalnaved/sensing-not-alerting)
- Nouyed et al. "Comparative Analysis of General-Purpose vs. Domain-Specific Multimodal Models for Diabetic Retinopathy Classification." *Diagnostics*, 2026. [doi:10.3390/diagnostics16101504](https://doi.org/10.3390/diagnostics16101504)
- Akhter, Nouyed et al. "Evolving performance of GPT models in dermoscopic diagnosis." *JAAD*, 2026. [doi:10.1016/j.jaad.2025.09.119](https://doi.org/10.1016/j.jaad.2025.09.119)
- Keplinger LE, Hu G. "Limitations of ChatGPT in dermatological image analysis: Toward broader and deeper evaluation." *J Invest Dermatol.* 2025;145(12):3187-3188.

---

## Funding

NSF 2125872 (D.A. Adjeroh, G. Hu).

---

## License

MIT License. If you use this framework, please cite the paper above.
