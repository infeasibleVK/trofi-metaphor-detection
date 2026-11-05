# Task 1: Classical Selectional-Preference Violation (SPV) Baseline

**Course:** CS60075 Natural Language Processing (Assignment II)  
**Project:** Project 11 — Metaphor Detection: Selectional-Preference Violation vs. Neural Token Classification  
**Dataset:** TroFi (Trope Finder)  
**Assigned Weightage:** 30 Marks  

---

## 1. Task Objective

Task 1 implements a fully classical, non-neural statistical metaphor detector on the **TroFi dataset** using Selectional Preference Violation (SPV) theory.

The objective is to:
1. Parse and preprocess the raw TroFi block structure into clean, reproducible train/validation/test splits (80/10/10).
2. Extract grammatical predicate-argument relations (subject and direct/prepositional object) for each occurrence of the 50 ambiguous target verbs.
3. Map argument nouns to WordNet lexicographer supersenses ($|C| = 26$).
4. Compute Resnik's (1993, 1997) information-theoretic **Selectional Preference Strength (SPS)** and **Selectional Association Scores**.
5. Optimize the classification decision threshold strictly on held-out **validation data** to maximize Metaphor F1.
6. Evaluate the frozen baseline on the **held-out test set** reporting Precision, Recall, F1 (metaphor class), Accuracy, Confusion Matrix, and POS breakdown.
7. Export standardized association and SPV scores (`data/task1/spv_scores.csv` and `data/task1_scores.csv`) for feature augmentation and point-biserial correlation analysis in **Task 3**.

---

## 2. Theoretical Grounding & SPV Intuition

In classical linguistics and computational semantics (Wilks 1975, 1978; Resnik 1993, 1997), a verb imposes semantic restrictions on its argument positions:
- For example, the verb *eat* strongly prefers arguments denoting physical sustenance (`noun.food`).
- The verb *digest* prefers food or nutritive objects in its literal sense.
- When *digest* takes an abstract, organizational direct object—as in *"the company digested its rival"*—the argument violates the expected semantic category. This **Selectional Preference Violation (SPV)** signals figurative or metaphorical language.

Task 1 formalizes this hypothesis purely using corpus statistics and WordNet semantic categories without neural parameters or gradient updates.

---

## 3. Mathematical Formulation

### 3.1 WordNet Semantic Classes (Supersenses)
Nouns in WordNet are partitioned into 26 broad lexicographer categories (supersenses):
$$C = \{\text{noun.act}, \text{noun.animal}, \text{noun.artifact}, \dots, \text{noun.substance}, \text{noun.time}\}$$

For an argument noun $w$ exhibiting polysemy, let $S(w)$ be its set of noun synsets in WordNet. Following Resnik's (1995, 1997) uniform sense-weighting formulation:
$$P(c \mid w) = \frac{\sum_{s \in S(w), \text{lexname}(s) = c} 1}{|S(w)|}$$

### 3.2 Relation-Specific Priors $P_R(c)$
In Resnik (1993, 1997), selectional preferences are explicitly defined with respect to an argument relation $R \in \{\text{obj}, \text{subj}\}$. Because subjects (predominantly animate agents, persons, organizations) and objects (predominantly concrete artifacts, substances, possessions, communicative abstractions) exhibit vastly different distributional baselines in natural language, using a single blended prior causes distortion. We therefore maintain separate priors for direct/prepositional objects and subjects, smoothed with Lidstone smoothing ($\alpha = 0.1$):
$$P_R(c) = \frac{\sum_{v'} \text{Count}_R(v', c) + \alpha}{\sum_{v'} \sum_{c'} (\text{Count}_R(v', c') + \alpha)}$$

### 3.3 Conditional Distribution $P_R(c \mid v)$
The probability that verb $v$ selects semantic category $c$ for argument position $R \in \{\text{obj}, \text{subj}\}$:
$$P_R(c \mid v) = \frac{\text{Count}_R(v, c) + \alpha}{\sum_{c'} (\text{Count}_R(v, c') + \alpha)}$$

### 3.4 Resnik's Selectional Preference Strength (SPS)
Resnik (1993, 1997) defines the Selectional Preference Strength $SPS_R(v)$ as the Kullback-Leibler (KL) divergence between the conditional argument distribution $P_R(c \mid v)$ and the relation-specific prior $P_R(c)$:
$$SPS_R(v) = D_{\text{KL}}(P_R(c \mid v) \parallel P_R(c)) = \sum_{c \in C} P_R(c \mid v) \log_2 \frac{P_R(c \mid v)}{P_R(c)}$$
- $SPS_R(v) \ge 0$: A value near $0$ indicates a verb with weak or generic preferences (takes any noun class).
- Higher $SPS_R(v)$ indicates a strongly selective verb that restricts its argument fillers to specific classes.

### 3.5 Per-Token Association Score
Following Step 3 of the assignment specification, the selectional association between predicate $v$ and an argument noun $w$ at relation $R$ with supersense distribution $P(c \mid w)$ is the expected Pointwise Mutual Information (PMI) log-ratio:
$$\text{Assoc}_R(v, w) = \sum_{c \in C} P(c \mid w) \log_2 \frac{P_R(c \mid v)}{P_R(c)}$$

**Interpretation:**
- $\text{Assoc}_R(v, w) > 0$: The argument class is *more frequent* with verb $v$ than expected under general English object/subject distributions $\implies$ argument satisfies selectional preferences $\implies$ **Literal**.
- $\text{Assoc}_R(v, w) < 0$: The argument class is *atypical / dispreferred* given verb $v \implies$ selectional preference violation $\implies$ **Metaphorical**.

### 3.6 SPV Violation Score and Multi-Argument Combination
The Selectional Preference Violation score is defined as the negated association score:
$$\text{SPV}(v, w) = -\text{Assoc}(v, w)$$
Higher $\text{SPV}$ indicates stronger selectional violation.

When both subject and object arguments are present, a violation in *either* slot signals metaphor:
$$\text{Assoc}_{\text{combined}} = \min\left(\text{Assoc}_{\text{obj}}(v, w_{\text{obj}}),\, \text{Assoc}_{\text{subj}}(v, w_{\text{subj}})\right)$$
$$\text{SPV}_{\text{combined}} = -\text{Assoc}_{\text{combined}} = \max\left(\text{SPV}_{\text{obj}},\, \text{SPV}_{\text{subj}}\right)$$

---

## 4. Pipeline Architecture & Implementation Details

```
TroFiExampleBase.txt
         │
         ▼
[preprocess.py] ──> Discard U, Map N->1 / L->0, Locate Target Verb
         │
         ├──> data/trofi_train.csv (80%, 2,989 rows)
         ├──> data/trofi_val.csv   (10%,   374 rows)
         └──> data/trofi_test.csv  (10%,   374 rows)
         │
         ▼
[spv_detector.py]
         ├── ArgumentExtractor (spaCy en_core_web_sm: dep parsing, passives, relative clauses)
         ├── WordNetSenseProfiler (WordNet 26 supersenses, uniform polysemy weighting)
         └── ResnikSPVModel (P(c), P(c|v), SPS(v), Association, SPV score)
         │
         ▼
[evaluate_task1.py]
         ├── 1. Fit SPV profiles on Train split only
         ├── 2. Grid-search optimal threshold tau* on Validation split (Maximize F1)
         ├── 3. Evaluate frozen model & tau* on Test split
         └── 4. Export data/task1/spv_scores.csv & data/task1_scores.csv for Task 3
```

### 4.1 Syntactic Argument Extraction Strategy
Using spaCy (`en_core_web_sm`), `ArgumentExtractor` identifies:
1. **Direct Subject & Object:**
   - Subject: `nsubj`, `csubj`.
   - Direct Object: `dobj`, `obj`.
2. **Passive Voice Handling:**
   - Detects `auxpass` / `nsubjpass`.
   - In passive constructions, the syntactic subject (`nsubjpass`) is semantically the *patient/object*.
   - The prepositional agent `by` $\to$ `pobj` is semantically the *agent/subject*.
3. **Relative Clause Antecedents:**
   - If the subject is a relative pronoun (`who`, `which`, `that`) in a relative clause (`relcl`), resolves the governing antecedent noun (e.g., *"assistance, which absorbs the budget"* $\to$ resolves antecedent *assistance* as subject).
4. **Clausal Complements & Control:**
   - If the target verb appears in an infinitival or participial complement (`xcomp`, `acl`), inherits the subject from the governing clause head.
5. **Prepositional Object Fallback:**
   - For intransitive or prepositional verbs (e.g. *pour into*, *plow into*, *strike at*), extracts the complement `pobj`.

### 4.2 Cold-Start and Smoothing Strategy
- **Lidstone Smoothing:** $\alpha = 0.1$ prevents zero probabilities and infinite log-ratios for unseen verb-class pairs.
- **Cold-Start Fallback:** When a sentence lacks an extractable argument or the argument is out-of-vocabulary, `cold_start_flag = 1`. Following Step 5 of the assignment specification, cold-start instances fall back to the majority class (Literal = 0) with a neutral score ($\text{SPV} = 0.0$).

### 4.3 Validation-Only Threshold Tuning (Zero Test Leakage)
- Threshold $\tau^*$ is selected strictly by grid search over 500 candidate thresholds on the **validation set** to maximize Metaphor F1:
  $$\tau^* = \arg\max_\tau F_1^{\text{val}}(\tau)$$
- The test set is evaluated exactly **once** using this frozen threshold. Test labels are never used during fitting or tuning.

---

## 5. Directory Structure & File Manifest

```text
code/task1/
├── preprocess.py       # TroFi raw parser, regex, and split generator
├── spv_detector.py     # Argument extractor, WordNet profiler, and Resnik SPV model
├── evaluate_task1.py   # Training, validation threshold tuning, and test evaluation
├── run_task1.sh        # One-click execution shell script
└── README.md           # Comprehensive methodology and execution documentation
```

---

## 6. Execution Instructions

Run all commands from the repository root inside the activated `.venv`:

### Option A: One-Click Execution
```bash
./code/task1/run_task1.sh
```

### Option B: Step-by-Step Execution

#### Step 1: Preprocess Dataset
```bash
python code/task1/preprocess.py \
  --data_path data/TroFiExampleBase.txt \
  --output_dir data/ \
  --seed 42
```

#### Step 2: Fit SPV Profiles, Tune Threshold on Val, and Evaluate on Test
```bash
python code/task1/evaluate_task1.py \
  --train_path data/trofi_train.csv \
  --val_path data/trofi_val.csv \
  --test_path data/trofi_test.csv \
  --output_scores data/task1/spv_scores.csv \
  --alpha 0.1 \
  --profile_source all
```

---

## 7. Experimental Results

### 7.1 Dataset Statistics
- Total lines parsed: **6,786**
- Verbs detected: **50 ambiguous verbs**
- Discarded unannotated (`U`) lines: **2,699**
- Retained clean labeled lines: **3,737** (Metaphorical `N`: 2,145, Literal `L`: 1,592)
- Splits (80/10/10 sentence-level, seed=42):
  - **Train:** 2,989 instances (1,716 Metaphor, 1,273 Literal — 57.4% positive)
  - **Val:** 374 instances (214 Metaphor, 160 Literal — 57.2% positive)
  - **Test:** 374 instances (215 Metaphor, 159 Literal — 57.5% positive)

### 7.2 Validation Tuning & Model Selection
Comparing candidate configurations on the validation split (`seed=42`, test set untouched):

| Configuration | Profile Source | Prior Type | Optimal $\tau^*$ | Val Metaphor $F_1$ | Val Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Shared Prior | `all` | Shared $P(c)$ | -1.6328 | 0.7358 | 63.90% |
| Shared Prior | `literal` | Shared $P(c)$ | -1.2039 | 0.7360 | 64.71% |
| **Relation-Specific** | `all` | $P_{\text{obj}}, P_{\text{subj}}$ | **-1.5403** | **0.7387** | **64.44%** |
| **Relation-Specific** | `literal` | $P_{\text{obj}}, P_{\text{subj}}$ | **-1.6710** | **0.7387** | **64.44%** |

Selected Configuration: **Relation-Specific Priors with `profile_source='all'`** (per Step 2 assignment instructions).

### 7.3 Held-Out Test Evaluation Results (Frozen $\tau^* = -1.5403$)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **61.76%** |
| **Metaphor Precision** | **0.6224** |
| **Metaphor Recall** | **0.8512** |
| **Metaphor F1 Score** | **0.7191** |
| **Macro F1 Score** | **0.5604** |

#### Confusion Matrix (Test Set, N=374)
| | Predicted Literal (0) | Predicted Metaphor (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Literal (0)** | 48 (TN) | 111 (FP) | 159 |
| **Actual Metaphor (1)** | 32 (FN) | 183 (TP) | 215 |

#### Classification Report
```
              precision    recall  f1-score   support

 Literal (0)       0.60      0.30      0.40       159
Metaphor (1)       0.62      0.85      0.72       215

    accuracy                           0.62       374
   macro avg       0.61      0.58      0.56       374
weighted avg       0.61      0.62      0.58       374
```

### 7.4 Part-of-Speech Breakdown (TroFi Ambiguous Verbs)
In TroFi, all target words are ambiguous verbs (`pos == VERB`). Breakdown by fine-grained Penn Treebank verbal inflection tags:

| POS Tag | Inflection Type | Support | Metaphor F1 | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| `VB` | Base form (e.g. *absorb*) | 108 | 0.7077 | 62.04% |
| `VBD` | Past tense (e.g. *absorbed*) | 74 | 0.7013 | 59.46% |
| `VBG` | Gerund/Present Participle (e.g. *absorbing*) | 82 | 0.7419 | 64.63% |
| `VBN` | Past Participle (e.g. *absorbed*) | 62 | 0.8172 | 72.58% |
| `VBZ` | 3rd person singular present (e.g. *absorbs*) | 14 | 0.6667 | 57.14% |
| `VBP` | Non-3rd person present (e.g. *absorb*) | 23 | 0.5517 | 43.48% |
| **Overall VERB** | **All verbal forms** | **363** | **0.7254** | **62.26%** |

---

## 8. Task 3 Integration & Output Schema

The pipeline exports three distinct, dedicated tables to serve Task 3:
1. `data/task1/spv_scores.csv`: Comprehensive table with all features, predictions, and metadata.
2. `data/task1_scores.csv`: Continuous SPV association scores and cold-start flags tailored for `code/task3/correlation.py`.
3. `data/task1_preds.csv`: Discrete predictions, gold labels, and correctness flags tailored for `code/task3/disagreement_analysis.py`.
4. Separate split files: `data/task1/spv_scores_train.csv`, `data/task1/spv_scores_val.csv`, and `data/task1/spv_scores_test.csv` for `code/task3/augmented_model.py`.

### Exported Column Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `sentence_id` | string | WSJ source identifier (e.g., `wsj03:2839`) |
| `sentence` | string | Tokenized sentence text |
| `target_verb` | string | Base lemma of the target verb (e.g., `absorb`) |
| `target_index` | int | 0-based token index of the target verb |
| `target_pos` | string | Universal POS tag (`VERB`) |
| `target_tag` | string | Penn Treebank POS tag (`VB`, `VBD`, `VBG`, etc.) |
| `gold_label` | int | Ground truth label (1 = Metaphor, 0 = Literal) |
| `spv_score` | float | Selectional violation score ($-\text{Assoc}$; higher $\implies$ violation) |
| `association_score` | float | Resnik association score ($\text{Assoc}$; higher $\implies$ typical/literal) |
| `task1_prediction` | int | Model prediction (1 = Metaphor, 0 = Literal) |
| `cold_start_flag` | int | 1 if no argument was found/scored (fallback used); 0 if scored |
| `sps_verb` | float | Resnik Selectional Preference Strength $SPS(v)$ |
| `subj_noun` | string | Extracted subject head lemma |
| `obj_noun` | string | Extracted object head lemma |
| `subj_score` | float / null | Raw association score for subject |
| `obj_score` | float / null | Raw association score for object |
| `is_passive` | int | 1 if passive construction, 0 otherwise |

---

## 9. Limitations & Academic Discussion

1. **Polysemy and Word Sense Disambiguation:**
   - In classical SPV without supervised sense disambiguation, nouns are assigned uniform weights across their WordNet synsets. For highly polysemous words (e.g., *head*, *fund*), this introduces slight noise into the supersense distribution.
2. **Missing Syntactic Arguments:**
   - In intransitive constructions or complex subordinate clauses, verbs may lack an overt subject or direct object. The pipeline resolves relative clause antecedents and prepositional complements, but 10–12% of sentences require cold-start fallback.
3. **Fixed Split vs. Cross-Validation:**
   - As explicitly mandated by the assignment, a fixed 80/10/10 sentence-level split is evaluated. While published literature (Gao et al. 2018; MelBERT 2020) uses 10-fold cross-validation per verb, our fixed split maintains strict reproducibility and fair comparison across Tasks 1, 2, and 3. This methodological deviation is documented in accordance with the project rubric.
