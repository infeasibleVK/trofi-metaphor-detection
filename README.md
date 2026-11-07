# CS60075 NLP Assignment II — Project 11: Metaphor Detection
### Selectional-Preference Violation vs. Neural Token Classification
**Dataset:** TroFi (Trope Finder)  
**Submission Deadline:** November 8, 2026  
**Team Size:** 3 Members  

---

## 📌 Project Overview

This project investigates whether neural metaphor detection aligns with linguistic theories of **Selectional Preference Violation (SPV)**. In classical linguistics, a metaphor occurs when a predicate violates the expected semantic category of its arguments (e.g., *“the company digested its rival”* — *digest* expects a food-type direct object, not an abstract organization).

In this project, your group evaluates this question across three key tasks on the **TroFi dataset**:
1. **Task 1 (Baseline — 30 Marks):** Classical statistical metaphor detector based on Resnik's Selectional Preference Strength and WordNet supersenses.
2. **Task 2 (Paper Reproduction — 30 Marks):** Neural BiLSTM token-level classifier combining GloVe and ELMo contextual representations (reproducing Gao et al., EMNLP 2018).
3. **Task 3 (Combine & Improve — 40 Marks):** Feature-augmented neural model fusing Task 1's SPV association scores, along with point-biserial correlation and disagreement subset analysis.

---

## 👥 3-Member Work Distribution & Timeline

| Member | Primary Role | Core Responsibilities |
| :--- | :--- | :--- |
| **Member A** | **Data & Task 1 Lead** | • Parse & clean raw TroFi dataset (discard `U` labels, locate target verbs).<br>• Generate reproducible 80/10/10 train/val/test splits.<br>• Dependency parsing & POS tagging via spaCy.<br>• Implement Resnik SPS & WordNet supersense profiling.<br>• Export association scores `task1/spv_scores.csv` for Task 3.<br>• Draft Task 1 methodology & results in the report. |
| **Member B** | **Task 2 (Neural) Lead** | • Implement 2-layer BiLSTM encoder (hidden size 300/dir, dropout 0.5).<br>• Implement target-word classification head (TroFi target verb only).<br>• Setup PyTorch training loop with Adam optimizer and validation early-stopping.<br>• Modify architecture to incorporate scalar SPV features + cold-start flags for Task 3.<br>• Draft Task 2 model reproduction and comparison with published numbers. |
| **Member C** | **Infra, Task 3 & Report Lead** | • Repository skeleton, environment management (`requirements.txt`).<br>• Embedding download & caching pipeline (GloVe 840B.300d + ELMo).<br>• Point-biserial correlation analysis (Task 1 SPV vs. Task 2 correctness).<br>• Disagreement subset analysis (Task 1 vs. Task 2 errors).<br>• Final report assembly (8–12 pages LaTeX), README documentation, and packaging. |

### 5-Day Milestone Timeline (Nov 3 – Nov 8)
- **Day 1 (Nov 3-4):** Setup environment, parse TroFi dataset, extract target verb indices, download GloVe/ELMo embeddings.
- **Day 2 (Nov 4-5):** Complete Task 1 (Resnik SPV baseline) + implement Task 2 BiLSTM architecture.
- **Day 3 (Nov 5-6):** Train Task 2 model, export SPV scores, compute point-biserial correlation, build Task 3 feature-augmented model.
- **Day 4 (Nov 6-7):** Run Task 3 training and evaluation, perform disagreement subset analysis, compile quantitative tables & qualitative examples.
- **Day 5 (Nov 7-8):** Finalize report (8–12 pages PDF), record 5–10 min walkthrough demo, prepare 10 min presentation slides, package archive.

---

## 🗂️ Project Directory Structure

```text
NLP_assignment_2/
├── README.md                          # Project documentation and run guide
├── requirements.txt                   # Pinned Python package dependencies
├── environment.yml                    # Conda environment definition
├── report.pdf                         # 8-12 page project report
├── data/                              # Local dataset folder (ignored from git/zip)
│   ├── TroFiExampleBase.txt           # Raw TroFi dataset
│   ├── trofi_train.csv                # 80% train split
│   ├── trofi_val.csv                  # 10% validation split
│   ├── trofi_test.csv                 # 10% test split
│   └── embeddings/                    # Cached GloVe and ELMo files
├── code/
│   ├── task1/
│   │   ├── preprocess.py              # TroFi parser, regex, and spaCy pipeline
│   │   ├── spv_detector.py            # WordNet supersenses, Resnik SPS, association scores
│   │   ├── evaluate_task1.py          # Threshold grid-search & evaluation
│   │   └── run_task1.sh (or .bat)     # One-click execution script
│   ├── task2/
│   │   ├── dataset.py                 # PyTorch Dataset for TroFi target verbs
│   │   ├── model.py                   # 2-layer BiLSTM + GloVe + ELMo
│   │   ├── train.py                   # Training loop with early stopping
│   │   ├── evaluate_task2.py          # Metric computation & POS breakdown
│   │   └── run_task2.sh (or .bat)     # One-click execution script
│   ├── task3/
│   │   ├── correlation.py             # Point-biserial correlation computation
│   │   ├── augmented_model.py         # BiLSTM with SPV scalar + cold-start feature
│   │   ├── disagreement_analysis.py   # Breakdown of disagreement subset
│   │   ├── evaluate_task3.py          # Comparative evaluation (Task 1 vs 2 vs 3)
│   │   └── run_task3.sh (or .bat)     # One-click execution script
│   └── utils/
│       ├── embeddings.py              # Helper to extract GloVe and ELMo vectors
│       ├── metrics.py                 # Precision, Recall, Macro-F1 calculations
│       └── seed.py                    # Global seedsetter (seed=42)
```

---

## ⚙️ Environment Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU recommended with CUDA 11.8+ / 12.1+

### Installation Steps

```bash
# 1. Create and activate a clean conda environment
conda create -n nlp_trofi python=3.10 -y
conda activate nlp_trofi

# 2. Install PyTorch matching your CUDA setup (or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install required libraries
pip install spacy nltk scikit-learn scipy pandas numpy allennlp allennlp-models matplotlib

# 4. Download spaCy model & NLTK data
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 📥 Data & Pretrained Weights Acquisition

### 1. TroFi Dataset
Download the original `TroFiExampleBase.txt`:
```bash
mkdir -p data
# From SFU mirror
curl -L https://raw.githubusercontent.com/sfu-natlang/trofi-metaphor-data/master/TroFiExampleBase.txt -o data/TroFiExampleBase.txt
```

### 2. Pretrained Embeddings
- **GloVe 840B.300d:**
  ```bash
  mkdir -p data/embeddings
  wget https://nlp.stanford.edu/data/glove.840B.300d.zip -P data/embeddings/
  unzip data/embeddings/glove.840B.300d.zip -d data/embeddings/
  ```
- **ELMo Weights:** Downloaded automatically through `allennlp` or cached directly in `data/embeddings/`.

---

## 🚀 Execution Instructions

### Run Task 1: Classical SPV Baseline
```bash
# Preprocess TroFi and generate splits (seed=42)
python code/task1/preprocess.py --data_path data/TroFiExampleBase.txt --output_dir data/

# Train SPV tables, tune threshold on validation, and evaluate on test
python code/task1/evaluate_task1.py --train_path data/trofi_train.csv --val_path data/trofi_val.csv --test_path data/trofi_test.csv
```

### Run Task 2: Neural BiLSTM Reproduction
```bash
# Train BiLSTM with GloVe + ELMo
python code/task2/train.py ^
  --train_path data/trofi_train.csv ^
  --val_path data/trofi_val.csv ^
  --glove_path data/embeddings/glove.840B.300d.txt ^
  --elmo_options data/embeddings/elmo_2x4096_1xhighway_options.json ^
  --elmo_weights data/embeddings/elmo_2x4096_1xhighway_weights.hdf5 ^
  --save_dir checkpoints/task2/

# Evaluate on test set
python code/task2/evaluate_task2.py ^
  --test_path data/trofi_test.csv ^
  --model_path checkpoints/task2/best_model.pt ^
  --glove_path data/embeddings/glove.840B.300d.txt ^
  --elmo_options data/embeddings/elmo_2x4096_1xhighway_options.json ^
  --elmo_weights data/embeddings/elmo_2x4096_1xhighway_weights.hdf5
```

The Task 2 loader consumes the CSV files generated by Task 1. It accepts
`sentence`, `verb`, `label`, and either `target_index`, `target_idx`, or
`verb_idx`; a serialized `tokens` column is optional. The model performs
target-word-only classification, as required for TroFi. On Windows, the
equivalent end-to-end command is:

```bat
code\task2\run_task2.bat ^
  data\embeddings\glove.840B.300d.txt ^
  data\embeddings\elmo_2x4096_1xhighway_options.json ^
  data\embeddings\elmo_2x4096_1xhighway_weights.hdf5
```

### Run Task 3: Combine & Improve (SPV-Augmented Neural Detector)
```bash
# 1. Compute Point-Biserial Correlation
python code/task3/correlation.py

# 2. Train Augmented BiLSTM with SPV feature
python code/task3/augmented_model.py ^
  --glove_path data/embeddings/glove.840B.300d.txt ^
  --elmo_options data/embeddings/elmo_2x4096_1xhighway_options.json ^
  --elmo_weights data/embeddings/elmo_2x4096_1xhighway_weights.hdf5

# 3. Disagreement analysis and comprehensive comparison
python code/task3/disagreement_analysis.py
python code/task3/evaluate_task3.py
```

Task 3 uses the split-specific Task 1 exports
`data/task1/spv_scores_{train,val,test}.csv` and the Task 2 export
`checkpoints/task2/test_predictions.csv`. See
[code/task3/README.md](code/task3/README.md) for the full architecture,
feature normalization policy, outputs, and interpretation guidance.

---

## 📊 Evaluation & Metrics Summary

All tasks report metrics for the **metaphor class**:
- **Precision (P), Recall (R), and F1 Score**
- **F1 by Part-of-Speech:** (For TroFi, all target words are ambiguous verbs)
- **Task 3 Specific Metrics:**
  - Point-biserial correlation: $r_{pb}$ between continuous SPV association scores and binary neural correctness.
  - Accuracy and F1 on the **disagreement subset** ($\text{Prediction}_{\text{Task 1}} \neq \text{Prediction}_{\text{Task 2}}$).

---

## ⚠️ Important Dataset & Experimental Notes

1. **TroFi Block Structure:**
   The file is split by verb header lines (e.g. `***absorb***`) into literal and nonliteral cluster sections. Each line has three tab-separated columns: `WSJ id`, `annotation label`, and `tokenized sentence`.
2. **Discard 'U' Labels:**
   Only lines labeled `L` (Literal = 0) and `N` (Nonliteral = 1) have ground truth. Discard all `U` (Unannotated) instances before splitting.
3. **Fixed 80/10/10 Split Deviation (Mandatory Report Mention):**
   Standard papers (e.g., Gao et al., MelBERT) evaluate TroFi using 10-fold cross-validation per verb. Per assignment instructions, this project uses a single fixed 80/10/10 sentence-level split with `seed=42`. You **must explicitly state this deviation in your report**.
4. **Target-Word Classification:**
   TroFi is a single target-word classification task, not full sequence labeling. Your output head must extract the hidden state corresponding to the target verb index.

---

## 📦 Submission Deliverables Checklist

- [ ] **Code Archive:** Zip containing `code/` (`task1/`, `task2/`, `task3/`, `utils/`), `README.md`, `requirements.txt`. (Do NOT include large checkpoints or raw GloVe vectors).
- [ ] **Report (`report.pdf`):** 8–12 pages covering methodology, quantitative tables, qualitative case studies, comparison against published numbers, and in-depth discussion.
- [ ] **Walkthrough Video:** 5–10 minute recorded walkthrough demonstrating reproducible execution.
- [ ] **Presentation:** 10-minute presentation ready for your assigned Project TA.
