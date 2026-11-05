#!/usr/bin/env python3
"""
Preprocess the TroFi (Trope Finder) dataset for Metaphor Detection.

Key responsibilities:
1. Parse raw TroFiExampleBase.txt block structure (delimited by ***verb*** headers).
2. Separate nonliteral (N) and literal (L) cluster sections; discard unannotated (U) records.
3. Map labels: N -> 1 (metaphorical), L -> 0 (literal).
4. Identify target verb token index within each sentence using lemma and morphological matching.
5. Generate reproducible 80/10/10 train/validation/test splits (sentence-level, seed=42).
6. Verify and export data/trofi_train.csv, data/trofi_val.csv, data/trofi_test.csv.

Reference:
- Birke & Sarkar (EACL 2006, NAACL-HLT 2007)
- Assignment II Project 11 Guidelines (CS60075)
"""

import os
import re
import argparse
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
from nltk.corpus import wordnet as wn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load spaCy model with dependency parser and tagger."""
    try:
        nlp = spacy.load(model_name)
        return nlp
    except Exception as e:
        logger.error(f"Failed to load spaCy model '{model_name}': {e}")
        raise


def locate_target_verb_in_doc(doc: Any, target_verb: str) -> Tuple[int, str]:
    """
    Locate target verb token index and surface string directly within a spaCy Doc.
    Guarantees target_idx corresponds to actual spaCy token indices.
    Removes unsafe prefix matching: relies strictly on morphological analysis,
    spaCy lemmatization, and WordNet morphy.
    
    Returns:
        (target_idx, target_token)
    """
    t_verb_lower = target_verb.lower()
    verb_candidates = []
    other_candidates = []
    
    for idx, tok in enumerate(doc):
        tok_lower = tok.text.lower()
        tok_lemma = tok.lemma_.lower()
        wn_lemma = wn.morphy(tok_lower, wn.VERB)
        
        # Exact match on spaCy lemma, WordNet morphy, or token text
        if tok_lemma == t_verb_lower or wn_lemma == t_verb_lower or tok_lower == t_verb_lower:
            if tok.pos_ in ("VERB", "AUX"):
                verb_candidates.append((idx, tok.text))
            else:
                other_candidates.append((idx, tok.text))
                
    if verb_candidates:
        return verb_candidates[0][0], verb_candidates[0][1]
    elif other_candidates:
        return other_candidates[0][0], other_candidates[0][1]

    return -1, ""


def parse_trofi_raw(file_path: str, nlp: Any) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Parse TroFiExampleBase.txt into a structured DataFrame.
    
    Returns:
        df: DataFrame with columns [sentence_id, target_verb, target_idx, target_token, label, sentence, cluster]
        stats: dictionary with parsing counts
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TroFi dataset not found at: {file_path}")

    records: List[Dict[str, Any]] = []
    current_verb: str = ""
    current_cluster: str = ""
    
    stats = {
        "total_lines_read": 0,
        "u_discarded": 0,
        "n_metaphor": 0,
        "l_literal": 0,
        "verbs_found": set(),
        "unmatched_verbs": 0
    }

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stats["total_lines_read"] += 1
            line_str = line.strip()
            if not line_str:
                continue

            # Check verb header: ***verb***
            if line_str.startswith("***") and not line_str.startswith("*****"):
                current_verb = line_str.strip("*").strip().lower()
                stats["verbs_found"].add(current_verb)
                continue
            
            # Check cluster section
            if line_str.startswith("*nonliteral cluster*"):
                current_cluster = "nonliteral"
                continue
            elif line_str.startswith("*literal cluster*"):
                current_cluster = "literal"
                continue
            elif line_str.startswith("*****"):
                continue

            # Data row: WSJ_ID \t LABEL \t SENTENCE
            parts = line_str.split("\t")
            if len(parts) < 3:
                continue

            wsj_id = parts[0].strip()
            label_str = parts[1].strip()
            sentence = "\t".join(parts[2:]).strip()

            # Discard U (unannotated)
            if label_str == "U":
                stats["u_discarded"] += 1
                continue
            elif label_str == "N":
                label = 1  # Metaphorical
                stats["n_metaphor"] += 1
            elif label_str == "L":
                label = 0  # Literal
                stats["l_literal"] += 1
            else:
                logger.warning(f"Unexpected label '{label_str}' in line: {line_str}")
                continue

            doc = nlp(sentence)
            target_idx, target_token = locate_target_verb_in_doc(doc, current_verb)
            if target_idx == -1:
                stats["unmatched_verbs"] += 1
                logger.warning(f"Target verb '{current_verb}' not located in: {sentence[:60]}...")

            records.append({
                "sentence_id": wsj_id,
                "target_verb": current_verb,
                "target_idx": target_idx,
                "target_token": target_token,
                "label": label,
                "sentence": sentence,
                "cluster": current_cluster
            })

    df = pd.DataFrame(records)
    stats["verbs_count"] = len(stats["verbs_found"])
    stats["total_retained"] = len(df)
    return df, stats


def split_dataset(
    df: pd.DataFrame,
    seed: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into 80% train, 10% validation, 10% test.
    Splits at sentence level across all 50 pooled verbs, with fixed seed.
    """
    strat = df["label"] if stratify else None
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=strat
    )

    strat_temp = temp_df["label"] if stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=strat_temp
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def verify_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Run data integrity and distribution checks across splits."""
    logger.info("=== Split Verification Checks ===")
    total = len(train_df) + len(val_df) + len(test_df)
    logger.info(f"Total instances: {total} (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_pos = (split["label"] == 1).sum()
        n_neg = (split["label"] == 0).sum()
        pct_pos = (n_pos / len(split)) * 100
        n_verbs = split["target_verb"].nunique()
        missing_idx = (split["target_idx"] == -1).sum()
        logger.info(
            f"[{name}] Total: {len(split)} | Metaphor (1): {n_pos} ({pct_pos:.1f}%) | "
            f"Literal (0): {n_neg} ({100 - pct_pos:.1f}%) | Distinct Verbs: {n_verbs} | Missing Target Idx: {missing_idx}"
        )

    # Check verb overlap
    train_verbs = set(train_df["target_verb"])
    val_verbs = set(val_df["target_verb"])
    test_verbs = set(test_df["target_verb"])
    logger.info(f"Verbs in Train: {len(train_verbs)}/50 | Val: {len(val_verbs)}/50 | Test: {len(test_verbs)}/50")


def main():
    parser = argparse.ArgumentParser(description="Preprocess TroFi dataset for Metaphor Detection (Task 1)")
    parser.add_argument("--data_path", type=str, default="data/TroFiExampleBase.txt", help="Path to raw TroFi file")
    parser.add_argument("--output_dir", type=str, default="data/", help="Output directory for generated CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--stratify", action="store_true", default=True, help="Stratify splits by label")
    parser.add_argument("--no_stratify", dest="stratify", action="store_false", help="Disable label stratification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading spaCy 'en_core_web_sm' model...")
    nlp = load_spacy_model("en_core_web_sm")

    logger.info(f"Parsing raw TroFi dataset from '{args.data_path}'...")
    df, stats = parse_trofi_raw(args.data_path, nlp)

    logger.info(f"Parsed {stats['total_lines_read']} lines:")
    logger.info(f"  - Verbs detected: {stats['verbs_count']}")
    logger.info(f"  - Discarded U records: {stats['u_discarded']}")
    logger.info(f"  - Retained N (Metaphorical=1): {stats['n_metaphor']}")
    logger.info(f"  - Retained L (Literal=0): {stats['l_literal']}")
    logger.info(f"  - Total clean records: {stats['total_retained']}")
    logger.info(f"  - Unmatched verb instances: {stats['unmatched_verbs']}")

    logger.info(f"Creating 80/10/10 train/val/test splits (seed={args.seed}, stratify={args.stratify})...")
    train_df, val_df, test_df = split_dataset(df, seed=args.seed, stratify=args.stratify)

    verify_splits(train_df, val_df, test_df)

    train_path = os.path.join(args.output_dir, "trofi_train.csv")
    val_path = os.path.join(args.output_dir, "trofi_val.csv")
    test_path = os.path.join(args.output_dir, "trofi_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved splits to:")
    logger.info(f"  Train: {train_path} ({len(train_df)} rows)")
    logger.info(f"  Val:   {val_path} ({len(val_df)} rows)")
    logger.info(f"  Test:  {test_path} ({len(test_df)} rows)")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
