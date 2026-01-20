# Roman-Urdu Sentiment Classification — Calibrated Weak-Supervision + Self-Training (Thesis Code)

This notebook contains the **Roman-Urdu (RUSA-19)** implementation of our thesis framework for **3-class sentiment classification** (**negative / neutral / positive**) under limited labeled data.

The pipeline follows a **weak-supervision → learned combiner → curriculum self-training → few-shot refinement** procedure.  
All outputs are stored locally in Colab under:

`/content/Thesis_RomanUrdu_SA`

---

## 1) Project Overview

### Goal
Build a strong Roman-Urdu sentiment classifier using **minimal gold supervision** by combining:
- Multiple **weak teachers** (zero-shot, translation-based, lexicon)
- A **learned combiner (stacking)** trained on a small gold set (few-shot)
- **Self-training** with confidence/vote-based curriculum and **soft targets**
- Optional final **few-shot refinement** for correction of pseudo-label noise

### Label Space
- `negative`
- `neutral`
- `positive`

---

## 2) Method Summary (Phases)

### Phase 0 — Setup + Cleaning
- Creates a consistent workspace under `/content/Thesis_RomanUrdu_SA`
- Loads raw CSVs, standardizes column names, normalizes labels
- Removes null/empty texts and duplicates
- Saves cleaned files under:  
  `/content/Thesis_RomanUrdu_SA/datasets/rusa19_clean/`

Outputs:
- `rusa19_train_clean.csv`
- `rusa19_dev_clean.csv`
- `rusa19_test_clean.csv`
- `rusa19_fewshot64_clean.csv` (if provided)
- `label_stats.json`

---

### Phase 1 — Baselines (for comparison only)
1) **Fully supervised baseline**: `xlm-roberta-base`
   - Trained on labeled TRAIN (gold)
   - Evaluated on gold DEV and gold TEST
   - Weighted cross-entropy used for mild class imbalance

2) **Zero-shot baseline**: `joeddav/xlm-roberta-large-xnli`
   - Uses candidate labels: negative/neutral/positive
   - Evaluated on gold DEV and gold TEST

Outputs under:  
`/content/Thesis_RomanUrdu_SA/outputs/phase1/`
- `supervised_xlmr_base/` (checkpoints)
- `supervised_dev_preds.csv`, `supervised_test_preds.csv`
- `zeroshot_dev_preds.csv`, `zeroshot_test_preds.csv`
- `phase1_summary.json`

> Note: Phase 1 is included as a reference baseline. The thesis framework’s core method is Phase 2 onward.

---

### Phase 2 — Weak Supervision (Pseudo-labels)
We generate pseudo-label distributions from **three independent weak sources**:

**A) XNLI Zero-shot Teacher**
- Model: `joeddav/xlm-roberta-large-xnli`

**B) Translation-based Teacher**
- Roman-Urdu → Urdu transliteration: `Mavkif/m2m100_rup_rur_to_ur`
- Urdu → English translation: `Helsinki-NLP/opus-mt-ur-en`
- English sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- If transliteration/MT fails, fallback is:
  `cardiffnlp/twitter-xlm-roberta-base-sentiment`

**C) Lexicon Heuristic**
- Uses `roman_urdu_lexicon.csv`
- Polarity scoring with negators/intensifiers/diminishers


For each example we compute:
- `pseudo_label` (argmax of aggregated probs)
- `pseudo_confidence` (max aggregated prob)
- `votes_agree` (how many sources agree with final label)

Outputs under:  
`/content/Thesis_RomanUrdu_SA/outputs/phase2/`
- `train_probs_xnli.npy`
- `train_probs_en.npy`
- `train_probs_lex.npy`
- `train_texts.csv`
- `rusa19_train_pseudo.csv`
- `phase2_summary.json`

---

### Phase 2.5 — Learned Combiner (Stacking + Calibration)
Phase 2.5 trains a **meta-model combiner** to learn how to best combine the 3 weak teachers.

- Meta-model: **Multinomial Logistic Regression**
- Trained on a gold calibration set (**few-shot only**, 64 per class)
- Features:
  - concatenated teacher probabilities (9 dims)
  - meta features (confidence/entropy/agree/length/negator/intensifier) (10 dims)
  - total ~19 dims
- Hyperparameter search via `GridSearchCV`
- Probability calibration: `CalibratedClassifierCV(method="isotonic")`

Then the trained combiner is applied to the full TRAIN pool to produce **stacked pseudo-labels**.

Important:
- If Phase 2 `.npy` caches exist, Phase 2.5 loads them (fast restart).
- It also performs a basic order check with `train_texts.csv` to reduce misalignment risk.

Outputs under:  
`/content/Thesis_RomanUrdu_SA/outputs/phase2_5/`
- `combiner.joblib`
- `feature_info.json`
- `dev_cv_results.json`
- `dev_probs_{xnli,en,lex}.npy`
- `pool_probs_{xnli,en,lex}.npy`
- `rusa19_train_pseudo_stacked.csv`  ✅ **used in Phase 3**
- `phase2_5_summary.json`

The stacked pseudo CSV contains:
- hard pseudo label + confidence + votes
- calibrated combiner probabilities: `p_negative, p_neutral, p_positive`
- optional per-source probabilities (for analysis)

---

### Phase 3 — Self-Training (Curriculum + Soft Targets) + Few-shot Refine

**Stage-A (Self-training on pseudo labels)**
- Model: `xlm-roberta-base`
- Cold-start by default (no Phase-1 warm-start)
- Uses the stacked pseudo-labels from Phase 2.5
- Uses **soft targets**:
  - If combiner probs exist (`p_negative`, `p_neutral`, `p_positive`), they are used.
  - Otherwise soft targets are synthesized from `(pseudo_label, pseudo_confidence)`.
- Uses **example weights** based on:
  - pseudo_confidence
  - votes_agree

**Curriculum training order**
1. HIGH: votes=3 OR (votes=2 and conf>=0.70)
2. MID:  votes>=2 and conf>=0.50
3. ALL:  all pseudo samples

Loss:
- If soft targets exist: KL-divergence to soft targets
- Otherwise: CE fallback with class weights (rare fallback)

Artifacts saved:
- `curriculum_high.csv`, `curriculum_mid.csv`, `curriculum_all.csv`
- Stage-A checkpoint and predictions

**Stage-B (Few-shot refine)**
- Initializes from Stage-A checkpoint
- Fine-tunes on gold few-shot set (64/class)
- Evaluates on few-shot eval split + gold test split

Outputs under:  
`/content/Thesis_RomanUrdu_SA/outputs/phase3/`
- `stageA_selftrain/`
- `stageB_fewshot_refine/`
- `phase3_summary.json`
- `curriculum_high.csv`, `curriculum_mid.csv`, `curriculum_all.csv`

---

## 3) Data Requirements

Place these files in `/content/` (Colab session) before running:

Required:
- `/content/rusa19_train.csv`
- `/content/rusa19_dev.csv`  (used in Phase 1 baseline only)
- `/content/rusa19_test.csv`
- `/content/rusa19_fewshot64.csv` (required for Phase 2.5 + Stage-B)
- `/content/roman_urdu_lexicon.csv` (recommended; Phase 2 & 2.5 lexicon teacher)

Expected columns (flexible):
- Text column: `Text`, `text`, `review`, `utterance`, etc.
- Label column: `Sentiment`, `sentiment`, `label`, `polarity`

Labels accepted:
- `negative / neutral / positive` or common numeric/short forms (-1/0/1 etc.)

---

## 4) Environment / Dependencies

This notebook was designed for Google Colab with GPU.

Core packages:
- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`, `numpy`
- `tqdm`
- `joblib`
- `evaluate` (installed explicitly)

Notes:
- We disable Weights & Biases logging via:
  - `WANDB_DISABLED=true`
  - `WANDB_MODE=disabled`

---

## 5) How to Run (Recommended Order)

1) **Setup + Cleaning**
   - Creates directories + saves cleaned splits

2) **Phase 1 (Baselines)** *(optional but recommended for reporting)*
   - Supervised + zero-shot baselines

3) **Phase 2 (Weak supervision)**
   - Generates pseudo labels and caches `.npy` probabilities

4) **Phase 2.5 (Stacked combiner)**
   - Trains combiner using FEWSHOT
   - Generates stacked pseudo pack CSV

5) **Phase 3 (Self-training + refine)**
   - Stage-A curriculum training
   - Stage-B few-shot refinement
   - Exports predictions and summary

---

## 6) Reproducibility Notes

We set:
- `SEED=42`
- `set_seed(SEED)` (Transformers)
- `torch.manual_seed`, `np.random.seed`, `random.seed`
- `torch.backends.cudnn.deterministic=True` and `benchmark=False` (where used)

Despite seeds, slight run-to-run variation can still happen due to:
- GPU nondeterminism in some ops
- data loader ordering
- mixed precision (fp16)

---

## 7) Key Output Files

Important:
- `outputs/phase2/rusa19_train_pseudo.csv`
- `outputs/phase2_5/rusa19_train_pseudo_stacked.csv`
- `outputs/phase3/phase3_summary.json`
- `outputs/phase3/stageA_selftrain/preds_test.csv`
- `outputs/phase3/stageB_fewshot_refine/preds_test.csv`

Baselines:
- `outputs/phase1/supervised_test_preds.csv`
- `outputs/phase1/zeroshot_test_preds.csv`

---

## 8) Model & Teacher Details

Student model:
- `xlm-roberta-base`

Weak teachers:
- XNLI: `joeddav/xlm-roberta-large-xnli`
- Transliteration: `Mavkif/m2m100_rup_rur_to_ur`
- MT: `Helsinki-NLP/opus-mt-ur-en`
- EN Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Fallback multilingual sentiment: `cardiffnlp/twitter-xlm-roberta-base-sentiment`

Combiner:
- `LogisticRegression` + `isotonic calibration`

---

## 9) Known Constraints / Practical Notes

- Translation/transliteration is computationally expensive. Phase 2 caches `.npy` outputs so that:
  - Phase 2.5 can reuse them without rerunning Phase 2.
- Phase 2.5 expects few-shot file to exist; otherwise it raises an error by design.
- In Phase 2.5 we do not generate pseudo-labels using a simple rule like majority vote or max-confidence across teachers. Instead, we train a learned meta-combiner (multinomial logistic regression) on the few-shot gold set using the teachers’ probability outputs (plus meta-features), then apply this trained and calibrated combiner (isotonic calibration) to produce the final stacked pseudo-label probabilities and labels for the full training pool.
- Lexicon file is optional; if missing, lexicon teacher becomes uniform (weaker).

---


# ============================================================
# NOTE — Running the same pipeline with other backbones
# ============================================================
 This notebook is written with XLM-R as the default backbone:
   MODEL_NAME = "xlm-roberta-base"

 To run the *exact same framework* with a different pretrained model,
 you only need to replace the model name wherever the backbone is defined
 (i.e., the string passed to AutoTokenizer.from_pretrained(...) and
  AutoModelForSequenceClassification.from_pretrained(...)).

# Recommended backbone strings (HuggingFace model IDs):
   1) XLM-T (Twitter XLM-R variant):        "cardiffnlp/twitter-xlm-roberta-base"
      - Use when you want a model more tuned to social-media / short-text style.

   2) mDeBERTa-v3 (Microsoft):              "microsoft/mdeberta-v3-base"
      - Strong multilingual encoder; often good transfer and robustness.

   3) MuRIL (Google, Indic + Urdu focus):   "google/muril-base-cased"
      - Often strong for Urdu/Hindi scripts and related languages.

   4) mBERT (Multilingual BERT):            "bert-base-multilingual-cased"
      - Classic baseline multilingual encoder (older but widely used).

 Minimal changes you typically make:
   - Update MODEL_NAME in the "Tokenizer" section:
       MODEL_NAME = "<one of the model IDs above>"

 the rest of the pipeline (pseudo labels, curriculum, soft targets,
 trainers, evaluation, saving) stays the same.