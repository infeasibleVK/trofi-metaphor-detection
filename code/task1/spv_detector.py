#!/usr/bin/env python3
"""
Selectional Preference Violation (SPV) Detector for Metaphor Detection.

Implements Resnik's (1993, 1997) Information-Theoretic Selectional Preference
Strength (SPS) and Selectional Association models using WordNet supersenses
and spaCy dependency parses.

Key Components:
1. ArgumentExtractor: Extracts grammatical subject and object relations
   from sentences using spaCy dependency parse, handling passives, relative
   clauses, compound nouns, and clausal complements.
2. WordNetSenseProfiler: Maps nouns to WordNet synsets and lexicographer
   supersense distributions P(c | w).
3. ResnikSPVModel:
   - Computes prior P(c) and conditional P(c | v) distributions over supersenses.
   - Computes Selectional Preference Strength:
     SPS(v) = D_KL(P(c | v) || P(c)) = sum_c P(c | v) log2(P(c | v) / P(c))
   - Computes association score:
     Assoc(v, c) = log2(P(c | v) / P(c))
   - Computes SPV violation score:
     SPV(v, args) = -Assoc(v, args)
   - Separates subject and object profiles.
   - Handles cold starts with principled fallbacks.

References:
- Resnik, P. (1993). Selection and Information: A Class-Based Approach to
  Lexical Relationships. PhD thesis, University of Pennsylvania.
- Resnik, P. (1997). Selectional Preference and Sense Disambiguation.
  ACL SIGLEX Workshop.
- Wilks, Y. (1978). Making Preferences More Active. Artificial Intelligence.
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import spacy
from nltk.corpus import wordnet as wn

logger = logging.getLogger(__name__)


# 26 standard WordNet noun lexicographer files (supersenses)
WORDNET_NOUN_SUPERSENSES: List[str] = [
    "noun.Tops",
    "noun.act",
    "noun.animal",
    "noun.artifact",
    "noun.attribute",
    "noun.body",
    "noun.cognition",
    "noun.communication",
    "noun.event",
    "noun.feeling",
    "noun.food",
    "noun.group",
    "noun.location",
    "noun.motive",
    "noun.object",
    "noun.person",
    "noun.phenomenon",
    "noun.plant",
    "noun.possession",
    "noun.process",
    "noun.quantity",
    "noun.relation",
    "noun.shape",
    "noun.state",
    "noun.substance",
    "noun.time",
]

SUPERSENSE_TO_INDEX: Dict[str, int] = {s: i for i, s in enumerate(WORDNET_NOUN_SUPERSENSES)}
NUM_SUPERSENSES: int = len(WORDNET_NOUN_SUPERSENSES)


class ArgumentExtractor:
    """
    Extracts semantic subject and object arguments governing a target verb.
    Handles active, passive, relative, and clausal constructions.
    """

    def __init__(self, nlp: Optional[Any] = None):
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def locate_verb_token(
        self,
        doc: Any,
        target_idx: int = -1,
        target_verb: str = ""
    ) -> Optional[Any]:
        """Locate the target verb token within a spaCy Doc."""
        # 1. Check index if provided
        if 0 <= target_idx < len(doc):
            tok = doc[target_idx]
            tok_lower = tok.text.lower()
            wn_lemma = wn.morphy(tok_lower, wn.VERB)
            if (
                tok.lemma_.lower() == target_verb.lower()
                or wn_lemma == target_verb.lower()
                or tok_lower == target_verb.lower()
            ):
                return tok

        # 2. Search by lemma or morphy
        candidates = []
        t_verb = target_verb.lower()
        for idx, tok in enumerate(doc):
            tok_lower = tok.text.lower()
            wn_lemma = wn.morphy(tok_lower, wn.VERB)
            if tok.lemma_.lower() == t_verb or wn_lemma == t_verb or tok_lower == t_verb:
                is_verb = tok.pos_ in ("VERB", "AUX")
                candidates.append((is_verb, idx, tok))

        if candidates:
            candidates.sort(key=lambda x: (not x[0], x[1]))
            return candidates[0][2]

        return None

    def extract_arguments(
        self,
        doc: Any,
        target_idx: int = -1,
        target_verb: str = ""
    ) -> Dict[str, Any]:
        """
        Extract subject and object head nouns for a target verb occurrence.
        
        Returns:
            dict containing:
                - 'target_token': spacy Token of verb
                - 'subj_token': spacy Token of subject head (or None)
                - 'obj_token': spacy Token of object head (or None)
                - 'subj_lemma': string lemma
                - 'obj_lemma': string lemma
                - 'is_passive': boolean
        """
        v_tok = self.locate_verb_token(doc, target_idx, target_verb)
        if v_tok is None:
            return {
                "target_token": None,
                "subj_token": None,
                "obj_token": None,
                "subj_lemma": None,
                "obj_lemma": None,
                "is_passive": False,
            }

        subj_tok = None
        obj_tok = None
        is_passive = False

        # Identify passive voice: auxpass or nsubjpass
        for child in v_tok.children:
            if child.dep_ in ("auxpass", "nsubjpass", "csubjpass"):
                is_passive = True
                break

        # Passives:
        # - Syntactic subject (nsubjpass) is the semantic patient / object
        # - Agent prepositional phrase ("by X") contains semantic agent / subject
        if is_passive:
            for child in v_tok.children:
                if child.dep_ in ("nsubjpass", "csubjpass"):
                    obj_tok = child
                elif child.dep_ in ("agent", "prep") and child.text.lower() == "by":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            subj_tok = gc
                            break
        else:
            # Active voice:
            # Subject: nsubj, csubj
            for child in v_tok.children:
                if child.dep_ in ("nsubj", "csubj"):
                    subj_tok = child
                    break
            # Object: dobj, obj
            for child in v_tok.children:
                if child.dep_ in ("dobj", "obj"):
                    obj_tok = child
                    break

        # Relative clause antecedent resolution:
        # e.g., "the company, which absorbs the cost" -> subject "which" refers to "company"
        if subj_tok is not None and subj_tok.text.lower() in ("that", "which", "who", "whose"):
            if v_tok.dep_ == "relcl" and v_tok.head.pos_ in ("NOUN", "PROPN"):
                subj_tok = v_tok.head

        # Control and raising verbs:
        # e.g., "The bank tried to absorb..." -> 'absorb' has dep 'xcomp' with no overt subject.
        # Inherit subject from governing clause head.
        if subj_tok is None and v_tok.dep_ in ("xcomp", "acl", "advcl"):
            head_verb = v_tok.head
            for child in head_verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj_tok = child
                    break

        # Prepositional object fallback for intransitive / prepositional verbs:
        # e.g., "pour into the market", "plow into a wall", "flow into..."
        if obj_tok is None:
            for child in v_tok.children:
                if child.dep_ == "prep" and child.text.lower() != "by":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            obj_tok = gc
                            break
                    if obj_tok is not None:
                        break

        # Clausal complement object fallback:
        # e.g., "absorb polluting liquids" -> 'polluting' is xcomp, 'liquids' is dobj of 'polluting'
        if obj_tok is None:
            for child in v_tok.children:
                if child.dep_ in ("xcomp", "ccomp"):
                    for gc in child.children:
                        if gc.dep_ in ("dobj", "obj"):
                            obj_tok = gc
                            break
                    if obj_tok is not None:
                        break

        subj_lemma = subj_tok.lemma_.lower() if subj_tok is not None else None
        obj_lemma = obj_tok.lemma_.lower() if obj_tok is not None else None

        return {
            "target_token": v_tok,
            "subj_token": subj_tok,
            "obj_token": obj_tok,
            "subj_lemma": subj_lemma,
            "obj_lemma": obj_lemma,
            "is_passive": is_passive,
        }


class WordNetSenseProfiler:
    """
    Maps nouns to WordNet lexicographer supersenses.
    Follows Resnik's (1995, 1997) uniform sense-weighting scheme across noun synsets.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, float]] = {}

    def get_supersense_distribution(self, noun: Optional[str]) -> Dict[str, float]:
        """
        Compute P(c | w) for a given noun across the 26 noun supersenses.
        Distributes mass uniformly across all noun synsets of the word.
        """
        if not noun or not isinstance(noun, str):
            return {}

        clean_noun = noun.strip().lower()
        if clean_noun in self._cache:
            return self._cache[clean_noun]

        # 1. Query noun synsets
        synsets = wn.synsets(clean_noun, pos=wn.NOUN)
        if not synsets:
            # Try morphy
            morph = wn.morphy(clean_noun, wn.NOUN)
            if morph:
                synsets = wn.synsets(morph, pos=wn.NOUN)
        if not synsets:
            # Query all synsets as fallback
            synsets = wn.synsets(clean_noun)

        if not synsets:
            self._cache[clean_noun] = {}
            return {}

        distribution: Dict[str, float] = {}
        weight = 1.0 / len(synsets)
        for s in synsets:
            lexname = s.lexname()
            if lexname in SUPERSENSE_TO_INDEX:
                distribution[lexname] = distribution.get(lexname, 0.0) + weight

        self._cache[clean_noun] = distribution
        return distribution

    def get_most_frequent_supersense(self, noun: Optional[str]) -> Optional[str]:
        """Return the supersense of the most frequent sense (sense 1 in WordNet)."""
        if not noun:
            return None
        clean_noun = noun.strip().lower()
        synsets = wn.synsets(clean_noun, pos=wn.NOUN)
        if not synsets:
            morph = wn.morphy(clean_noun, wn.NOUN)
            if morph:
                synsets = wn.synsets(morph, pos=wn.NOUN)
        if not synsets:
            synsets = wn.synsets(clean_noun)
        if synsets:
            lexname = synsets[0].lexname()
            if lexname in SUPERSENSE_TO_INDEX:
                return lexname
        return None


class ResnikSPVModel:
    """
    Classical Selectional Preference Violation Model based on Resnik (1993, 1997).

    Estimates:
    - Prior distribution: P(c)
    - Posterior distributions: P_obj(c | v), P_subj(c | v), and combined P(c | v)
    - Selectional Preference Strength: SPS(v) = D_KL(P(c | v) || P(c))
    - Association score: Assoc(v, c) = log2( P(c | v) / P(c) )
    - Selectional Preference Violation: SPV(v, args) = -Assoc(v, args)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        profile_source: str = "all",
        prior_type: str = "relation_specific"
    ):
        """
        Args:
            alpha: Lidstone smoothing pseudo-count for probabilities.
            profile_source: 'all' to build profiles from all training sentences,
                            or 'literal' to build from literal sentences only.
            prior_type: 'relation_specific' to compute separate priors P_obj(c) and
                        P_subj(c) per Resnik (1997), or 'shared' for a single prior P(c).
        """
        self.alpha = alpha
        self.profile_source = profile_source
        self.prior_type = prior_type
        self.profiler = WordNetSenseProfiler()

        # Count tables
        self.obj_counts: Dict[str, np.ndarray] = {}
        self.subj_counts: Dict[str, np.ndarray] = {}
        self.total_obj_counts: np.ndarray = np.zeros(NUM_SUPERSENSES)
        self.total_subj_counts: np.ndarray = np.zeros(NUM_SUPERSENSES)
        self.all_counts: np.ndarray = np.zeros(NUM_SUPERSENSES)

        # Fitted probability distributions and statistics
        self.p_c: Optional[np.ndarray] = None
        self.p_obj_prior: Optional[np.ndarray] = None
        self.p_subj_prior: Optional[np.ndarray] = None
        self.sps_scores: Dict[str, float] = {}
        self.sps_obj_scores: Dict[str, float] = {}
        self.sps_subj_scores: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, training_records: List[Dict[str, Any]]):
        """
        Fit selectional-preference count tables on the training split.

        Args:
            training_records: list of dictionaries, each with:
                - 'target_verb': str
                - 'subj_lemma': Optional[str]
                - 'obj_lemma': Optional[str]
                - 'label': int (0=literal, 1=metaphorical)
        """
        logger.info(
            f"Fitting Resnik SPV Model on {len(training_records)} training records "
            f"(profile_source='{self.profile_source}', prior_type='{self.prior_type}', alpha={self.alpha})..."
        )

        self.obj_counts.clear()
        self.subj_counts.clear()
        self.total_obj_counts = np.zeros(NUM_SUPERSENSES)
        self.total_subj_counts = np.zeros(NUM_SUPERSENSES)
        self.all_counts = np.zeros(NUM_SUPERSENSES)

        used_count = 0
        for rec in training_records:
            label = rec.get("label", 0)
            if self.profile_source == "literal" and label == 1:
                continue

            v = rec["target_verb"].lower()
            if v not in self.obj_counts:
                self.obj_counts[v] = np.zeros(NUM_SUPERSENSES)
                self.subj_counts[v] = np.zeros(NUM_SUPERSENSES)

            # Object filler
            obj_lemma = rec.get("obj_lemma")
            if obj_lemma:
                ss_obj = self.profiler.get_supersense_distribution(obj_lemma)
                for ss, weight in ss_obj.items():
                    idx = SUPERSENSE_TO_INDEX[ss]
                    self.obj_counts[v][idx] += weight
                    self.total_obj_counts[idx] += weight
                    self.all_counts[idx] += weight

            # Subject filler
            subj_lemma = rec.get("subj_lemma")
            if subj_lemma:
                ss_subj = self.profiler.get_supersense_distribution(subj_lemma)
                for ss, weight in ss_subj.items():
                    idx = SUPERSENSE_TO_INDEX[ss]
                    self.subj_counts[v][idx] += weight
                    self.total_subj_counts[idx] += weight
                    self.all_counts[idx] += weight

            used_count += 1

        # Estimate smoothed prior P(c)
        total_class_counts = np.sum(self.all_counts)
        self.p_c = (self.all_counts + self.alpha) / (total_class_counts + self.alpha * NUM_SUPERSENSES)

        # Estimate relation-specific priors P_obj(c) and P_subj(c)
        tot_obj = np.sum(self.total_obj_counts)
        self.p_obj_prior = (self.total_obj_counts + self.alpha) / (tot_obj + self.alpha * NUM_SUPERSENSES)

        tot_subj = np.sum(self.total_subj_counts)
        self.p_subj_prior = (self.total_subj_counts + self.alpha) / (tot_subj + self.alpha * NUM_SUPERSENSES)

        # Estimate SPS(v) for each observed verb
        for v in self.obj_counts:
            # Combined verb count vector for general verb SPS
            comb_v = self.obj_counts[v] + self.subj_counts[v]
            total_v = np.sum(comb_v)
            p_c_v = (comb_v + self.alpha) / (total_v + self.alpha * NUM_SUPERSENSES)
            kl = np.sum(p_c_v * np.log2(p_c_v / self.p_c))
            self.sps_scores[v] = max(0.0, float(kl))

            # Relation-specific SPS
            tot_v_obj = np.sum(self.obj_counts[v])
            if tot_v_obj > 0:
                p_c_v_obj = (self.obj_counts[v] + self.alpha) / (tot_v_obj + self.alpha * NUM_SUPERSENSES)
                prior_obj = self.p_obj_prior if self.prior_type == "relation_specific" else self.p_c
                self.sps_obj_scores[v] = max(0.0, float(np.sum(p_c_v_obj * np.log2(p_c_v_obj / prior_obj))))
            else:
                self.sps_obj_scores[v] = 0.0

            tot_v_subj = np.sum(self.subj_counts[v])
            if tot_v_subj > 0:
                p_c_v_subj = (self.subj_counts[v] + self.alpha) / (tot_v_subj + self.alpha * NUM_SUPERSENSES)
                prior_subj = self.p_subj_prior if self.prior_type == "relation_specific" else self.p_c
                self.sps_subj_scores[v] = max(0.0, float(np.sum(p_c_v_subj * np.log2(p_c_v_subj / prior_subj))))
            else:
                self.sps_subj_scores[v] = 0.0

        self.is_fitted = True
        logger.info(
            f"Fitted SPV profiles for {len(self.obj_counts)} verbs from {used_count} records. "
            f"Mean Verb SPS: {np.mean(list(self.sps_scores.values())):.4f}"
        )

    def compute_argument_association(
        self,
        verb: str,
        noun: Optional[str],
        relation: str = "obj"
    ) -> Optional[float]:
        """
        Compute association score Assoc(v, w) for a given verb and argument noun:
        Assoc(v, w) = sum_c P(c | w) * log2( P(c | v) / P(c) )
        """
        if not self.is_fitted or not noun:
            return None

        verb = verb.lower()
        if relation == "obj":
            counts = self.obj_counts.get(verb)
            p_prior = self.p_obj_prior if self.prior_type == "relation_specific" else self.p_c
        else:
            counts = self.subj_counts.get(verb)
            p_prior = self.p_subj_prior if self.prior_type == "relation_specific" else self.p_c

        if counts is None:
            # Unseen verb: posterior equals prior -> assoc = 0.0
            return 0.0

        ss_dist = self.profiler.get_supersense_distribution(noun)
        if not ss_dist:
            return None

        total_v = np.sum(counts)
        p_c_v = (counts + self.alpha) / (total_v + self.alpha * NUM_SUPERSENSES)

        assoc = 0.0
        for ss, w in ss_dist.items():
            idx = SUPERSENSE_TO_INDEX[ss]
            ratio = p_c_v[idx] / p_prior[idx]
            assoc += w * math.log2(ratio)

        return assoc

    def score_instance(
        self,
        verb: str,
        subj_lemma: Optional[str] = None,
        obj_lemma: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute SPV features and association scores for a single predicate occurrence.

        Returns:
            dict containing:
                - 'association_score': float (positive = typical, negative = violation)
                - 'spv_score': float (-association_score, positive = violation/metaphor)
                - 'sps_verb': float (Selectional Preference Strength of verb)
                - 'subj_score': Optional[float]
                - 'obj_score': Optional[float]
                - 'cold_start_flag': int (1 if no argument found, 0 otherwise)
                - 'dominant_relation': str ('obj', 'subj', 'both', 'none')
        """
        verb = verb.lower()
        sps_v = self.sps_scores.get(verb, 0.0)

        obj_assoc = self.compute_argument_association(verb, obj_lemma, relation="obj")
        subj_assoc = self.compute_argument_association(verb, subj_lemma, relation="subj")

        scores = []
        if obj_assoc is not None:
            scores.append(("obj", obj_assoc))
        if subj_assoc is not None:
            scores.append(("subj", subj_assoc))

        if not scores:
            # Cold start: no argument found or out-of-vocabulary noun
            return {
                "association_score": 0.0,
                "spv_score": 0.0,
                "sps_verb": sps_v,
                "subj_score": None,
                "obj_score": None,
                "cold_start_flag": 1,
                "dominant_relation": "none",
            }

        # If both arguments exist:
        # A predicate is metaphorical if EITHER argument violates selectional preference.
        # Therefore, the combined association score is min(assoc), which corresponds
        # to the maximum selectional violation: spv = -min(assoc) = max(violation).
        best_rel, best_assoc = min(scores, key=lambda x: x[1])

        return {
            "association_score": float(best_assoc),
            "spv_score": float(-best_assoc),
            "sps_verb": sps_v,
            "subj_score": subj_assoc,
            "obj_score": obj_assoc,
            "cold_start_flag": 0,
            "dominant_relation": best_rel if len(scores) == 1 else "both",
        }
