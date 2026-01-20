# README — Calibrated Weak-Supervision + Curriculum Self-Training for Urdu Sentiment (SAU-18)

## Project Overview
This notebook implements the Urdu thesis framework for **3-class sentiment classification** (**negative / neutral / positive**) under **minimal labeled supervision**. The core method combines:

1. **Weak supervision (Phase 2):** Multiple noisy teachers assign pseudo-labels to the training pool.
2. **Few-shot calibration & stacking (Phase 2.5):** A small **few-shot gold** set calibrates and combines teachers via **Logistic Regression stacking + isotonic calibration**, producing calibrated **pseudo-probabilities**.
3. **Curriculum self-training (Phase 3 / Stage-A):** A Transformer model is trained on calibrated pseudo-labels using **soft targets** and **confidence/consensus-based weights**, progressing through **HI → MID → ALL**.
4. **Optional few-shot refinement (Phase 3B / Stage-B):** The Stage-A checkpoint is refined on the few-shot gold set (still no dev usage).

---

**Important (thesis framework rule):**
- Only **train + test + fewshot** are used (fewshot is mandatory).
- Test is **never** used for training or selection; it is final evaluation only.

A separate **Phase-1 baseline** section exists for comparison (zero-shot + fully supervised baseline). That baseline may use dev for early stopping, but it is **not part of the core framework pipeline**.

---

## Dataset Requirements

### Required files (in `/content/` when running on Colab)
Place the following CSV files in the Colab working directory:

- `sau18_train.csv`
- `sau18_test.csv`
- `sau18_fewshot_64_per_class.csv`  ✅ **required**

> Note: A `sau18_dev.csv` is used only in Phase-1 baseline experiments.  
> The **framework pipeline** does **not** use dev.

### Expected columns
Each CSV should contain at minimum:

- `text` (or a close variant such as `review`, `utterance`)
- `sentiment` or `label` (for fewshot and test gold labels)

The code includes **robust column detection** and normalization for:
- `pos/neg/neu`, `+/-/0`, numeric encodings, etc.

---

## Environment Setup

### Recommended runtime
- **Google Colab** with GPU enabled (T4 is usually enough).

### Core libraries
The notebook uses:
- `transformers`
- `datasets` (Phase-1 baseline)
- `torch`
- `scikit-learn`
- `evaluate`
- `joblib`
- `pandas`, `numpy`, `tqdm`

We already install:
```bash
pip install evaluate
```

---

## Output Structure (Single Source of Truth)

All artifacts are written under:

```
/content/outputs/
  data/
  phase1/
  phase2/
  phase2_5_stacked_cal/
  phase3_noextra/
  phase3_noextra_refine/
```

### Key output files
#### Phase 2 (Weak supervision)
- `outputs/phase2/sau18_train_pseudo.csv`  
  (pseudo labels + confidences per labeler + ensemble vote)

#### Phase 2.5 (Calibrated stacking)
- `outputs/phase2_5_stacked_cal/train_pseudo_stacked_cal.csv`
- `outputs/phase2_5_stacked_cal/combiner.joblib`
- `outputs/phase2_5_stacked_cal/meta.json`

#### Phase 3 (Stage-A curriculum training)
- `outputs/phase3_noextra/stageA_cal_soft/all/checkpoint-best/`
- `outputs/phase3_noextra/final_eval/preds_test.csv`
- `outputs/phase3_noextra/manifest.json`
- Curriculum CSVs:
  - `curriculum_hi.csv`
  - `curriculum_mid.csv`
  - `curriculum_all.csv`

#### Phase 3B (Stage-A+B few-shot refine)
- `outputs/phase3_noextra_refine/checkpoint-best/`
- `outputs/phase3_noextra_refine/preds_test.csv`
- `outputs/phase3_noextra_refine/manifest.json`

---

## Pipeline Walkthrough (Phases)

## Phase 1 — Baselines (Comparison Only)
This section is for comparison and reporting.

### 1A) Zero-shot baseline (XNLI)
- Model: `joeddav/xlm-roberta-large-xnli`
- Uses zero-shot classification on DEV/TEST (baseline comparison)

### 1B) Fully supervised baseline (XLM-R)
- Model: `xlm-roberta-base`
- Trained on full labeled train set, early stopping typically uses dev  
  (again: baseline only, not core framework)

Outputs saved in:
- `outputs/phase1/`

---

## Thesis Framework Pipeline (Core Method)

## Splits Resolver
Purpose:
- Ensures **train/test/fewshot** exist under:
  - `outputs/data/train.csv`
  - `outputs/data/test.csv`
  - `outputs/data/fewshot.csv`

Behavior:
- If the files already exist in `/content/`, it symlinks them.
- If the in-memory DFs exist, it writes them.
- If fewshot file is missing → raises error.

---

## Phase 2 — Weak Supervision (Pseudo-Labeling)
Goal:
- Produce pseudo labels for **TRAIN texts** using 3 independent labelers.

Labelers:
1. **XNLI zero-shot**  
   `joeddav/xlm-roberta-large-xnli`

2. **Translate Urdu → English + English sentiment model**  
   - Translator: `Helsinki-NLP/opus-mt-ur-en`
   - English sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`

3. **Lexicon heuristic**  
   - Small seed lexicon built in (optional file: `/content/urdu_lexicon.csv`)

Aggregation:
- Weighted confidence vote:
  ```python
  W = {"xnli": 1.0, "en": 1.0, "lex": 0.6}
  ```
- Produces:
  - `pseudo_label`
  - `pseudo_confidence`
  - `votes_agree` (0–3)

Saved as:
- `outputs/phase2/sau18_train_pseudo.csv`

---

## Phase 2.5 — Few-Shot Calibrated Stacking (Combiner)
Goal:
- Convert noisy pseudo-labels into **calibrated pseudo-probabilities**.

Gold source:
- **fewshot only** (`outputs/data/fewshot.csv`)

Method:
1. Build features from labelers (each contributes a probability triplet).
2. Fit **Logistic Regression stacking** (multinomial).
3. Apply **isotonic calibration** via `CalibratedClassifierCV`.
4. Predict calibrated probabilities for all train pseudo rows:
   - `pseudo_prob_negative`
   - `pseudo_prob_neutral`
   - `pseudo_prob_positive`

Outputs:
- `outputs/phase2_5_stacked_cal/train_pseudo_stacked_cal.csv`
- `combiner.joblib` (saved combiner)

**Note:** This is where the framework becomes “calibrated”: Phase-3 can train on **soft targets** rather than hard pseudo labels.

---

## Phase 3 — Stage-A Curriculum Self-Training (No-Extra-Data)
Goal:
- Train `xlm-roberta-base` using:
  - **soft targets** from Phase 2.5 (preferred)
  - **row weights** based on confidence & agreement
  - **curriculum**: HI → MID → ALL

### No-Extra-Data rule
- No back-translation.
- No augmentations.
- No adding new rows.
- Only reweighting.

### Row weight policy (light text cues)
Weights combine:
- pseudo confidence (higher = more weight)
- labeler agreement (votes_agree)
- simple Urdu cues:
  - negators (e.g., "نہیں")
  - intensifiers (e.g., "بہت")

### Curriculum bins
- **HI**: very high agreement/confidence
- **MID**: moderate agreement/confidence
- **ALL**: full pseudo dataset

### Training objective
Custom Trainer:
- Uses **KL divergence** between model distribution and soft targets:
  - Weighted KL for training
  - Standard CE for evaluation

Evaluation set for training selection:
- **fewshot** (`dsv_fs`) used for early stopping & best checkpoint selection

Final evaluation:
- Gold **TEST** only.
- Saves predictions to:
  - `outputs/phase3_noextra/final_eval/preds_test.csv`

---

## Phase 3B — Optional Few-Shot Refinement (Stage-A+B)
Goal:
- Fine-tune Stage-A checkpoint on fewshot gold labels only.

Training:
- Start from Stage-A best checkpoint.
- Train on fewshot for a few epochs.
- Early stopping uses the same fewshot set.

Final evaluation:
- TEST only
- Predictions saved to:
  - `outputs/phase3_noextra_refine/preds_test.csv`

---

## How to Run (Recommended Order)

### A) Baselines (Optional)
1. Phase-1 zero-shot + supervised baseline (for thesis comparison)

### B) Framework Pipeline (Core)
Run these in order:
1. **Outputs bootstrap**
2. **Splits resolver** (must succeed with fewshot)
3. **Aliases** (load df_train, df_test, df_fewshot)
4. **Phase 2** → produces pseudo CSV
5. **Phase 2.5** → produces calibrated pseudo probabilities
6. **Phase 3 (Stage-A)** → curriculum training + test preds
7. **Phase 3B (Stage-A+B)** → optional refinement + test preds

---

## Key Design Choices (Why These Steps)
- Weak supervision reduces dependence on large annotated datasets.
- Fewshot stacking calibrates noisy teacher outputs using minimal gold labels.
- Soft targets preserve uncertainty and help stabilize training.
- Curriculum reduces the effect of noisy pseudo labels early in training.
- No-extra-data rule keeps the method aligned with strict resource constraints.

---

## Reproducibility Notes
- Global seed is set (`SEED=42`) for:
  - Python `random`
  - NumPy
  - PyTorch

Despite seeds, slight run-to-run variation can still happen due to:
- GPU nondeterminism in some ops
- data loader ordering
- mixed precision (fp16)

---

## Troubleshooting

### 1) Fewshot required (no dev fallback)
If you see an error like:
> `Fewshot file is required ...`

Fix:
- Upload `sau18_fewshot_64_per_class.csv` into `/content/`
or ensure:
- `outputs/data/fewshot.csv` exists



### 2) Slow Phase 2 translation
- Translation + EN sentiment is compute-heavy.
- Reduce dataset size via `PSEUDO_MAX` for quick tests.

---

## NOTE: Running the same framework with other backbones

This notebook is written with **XLM-R** as the default backbone:

- `MODEL_NAME = "xlm-roberta-base"`

To run the **same pipeline** (Phase-1 / Phase-2 / Phase-3 / Phase-3B) with a different multilingual model, you generally only need to replace the HuggingFace model identifier wherever `MODEL_NAME` (and `tokenizer/model .from_pretrained`) is used.

### Recommended alternatives (HuggingFace model names)

1. **XLM-T**
   - `MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base"`  *(XLM-T base)*

2. **mDeBERTa v3 base**
   - `MODEL_NAME = "microsoft/mdeberta-v3-base"`

3. **MuRIL (Urdu/Hindi focused)**
   - `MODEL_NAME = "google/muril-base-cased"`

4. **Multilingual BERT**
   - `MODEL_NAME = "bert-base-multilingual-cased"`

### What to update (minimal)

- Replace `MODEL_NAME` in the config section(s)
- Ensure:
  - `tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)`
  - `model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, ...)`
