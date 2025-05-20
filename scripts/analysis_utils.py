#!/usr/bin/env python3
"""
Utilities for model evaluation and analysis.
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from .utils import TextNormalizer, NormalizationForm

logger = logging.getLogger(__name__)

class LemmaEvaluator:
    """Evaluates lemmatization performance across multiple models."""
    
    def __init__(self, models: List[spacy.language.Language], model_names: Optional[List[str]] = None, norm_method: Optional[str] = None):
        """
        Initialize evaluator with models to compare.
        
        Args:
            models: List of spaCy models to evaluate
            model_names: Optional list of model names for reporting
            norm_method: Optional normalization method
        """
        self.models = models
        self.model_names = model_names or [f'Model {i+1}' for i in range(len(models))]
        self.norm_method = norm_method
        
        if self.norm_method is None:
            warnings.warn("Normalization method not specified. Text may not be normalized correctly.", UserWarning)

    def evaluate_lemmas(self, 
                       docs: List[spacy.tokens.Doc],
                       evaluate_lemma: bool = True,
                       evaluate_pos: bool = False,
                       evaluate_morph: bool = False) -> pd.DataFrame:
        """
        Evaluate lemmatization performance across models.
        
        Args:
            docs: List of spaCy Docs with gold annotations
            evaluate_lemma: Whether to evaluate lemmatization
            evaluate_pos: Whether to evaluate POS tagging
            evaluate_morph: Whether to evaluate morphological analysis
            
        Returns:
            DataFrame with evaluation results
        """
        data = []
        skipped_tokens = {
            'total_tokens': 0,
            'empty_gold_lemma': 0,
            'empty_gold_pos': 0,
            'empty_gold_tag': 0,
            'empty_gold_morph': 0,
            'token_length_mismatch': 0
        }

        for doc in tqdm(docs, desc="Processing documents"):
            doc_text = doc.text
            predicted_docs = [model(doc_text) for model in self.models]

            for token_idx, token in enumerate(doc):
                skipped_tokens['total_tokens'] += 1
                
                # Check if token should be skipped
                skip_token = False
                if evaluate_lemma and not token.lemma_.strip():
                    skipped_tokens['empty_gold_lemma'] += 1
                    skip_token = True
                if evaluate_pos:
                    if not token.pos_.strip():
                        skipped_tokens['empty_gold_pos'] += 1
                        skip_token = True
                    if not token.tag_.strip():
                        skipped_tokens['empty_gold_tag'] += 1
                        skip_token = True
                if evaluate_morph:
                    morph_dict = token.morph.to_dict()
                    if not morph_dict or all(not value for value in morph_dict.values()):
                        skipped_tokens['empty_gold_morph'] += 1
                        skip_token = True
                
                if skip_token:
                    continue

                # Collect token data
                token_data = {"Text": doc.text, "Token": token.text}
                
                if evaluate_lemma:
                    token_data["Gold Lemma"] = token.lemma_
                    for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                        if token_idx < len(predicted_doc):
                            token_data[f"{model_name} Lemma"] = predicted_doc[token_idx].lemma_
                        else:
                            token_data[f"{model_name} Lemma"] = "N/A"
                            skipped_tokens['token_length_mismatch'] += 1

                if evaluate_pos:
                    token_data["Gold POS"] = token.pos_
                    token_data["Gold TAG"] = token.tag_
                    for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                        if token_idx < len(predicted_doc):
                            token_data[f"{model_name} POS"] = predicted_doc[token_idx].pos_
                            token_data[f"{model_name} TAG"] = predicted_doc[token_idx].tag_
                        else:
                            token_data[f"{model_name} POS"] = "N/A"
                            token_data[f"{model_name} TAG"] = "N/A"

                if evaluate_morph:
                    token_data["Gold Morph"] = str(token.morph.to_dict())
                    for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                        if token_idx < len(predicted_doc):
                            token_data[f"{model_name} Morph"] = str(predicted_doc[token_idx].morph.to_dict())
                        else:
                            token_data[f"{model_name} Morph"] = "N/A"

                data.append(token_data)

        # Print statistics
        self._print_statistics(skipped_tokens)
        
        # Calculate metrics
        df = pd.DataFrame(data)
        metrics = self._calculate_metrics(df, evaluate_lemma, evaluate_pos, evaluate_morph)
        logger.info("\nEvaluation Metrics:")
        logger.info(metrics.to_string(index=False))
        
        return df

    def _print_statistics(self, skipped_tokens: Dict[str, int]) -> None:
        """Print token skipping statistics."""
        total = skipped_tokens['total_tokens']
        logger.info("\nToken Skipping Statistics:")
        logger.info(f"Total tokens processed: {total}")
        for key, value in skipped_tokens.items():
            if key != 'total_tokens':
                logger.info(f"{key}: {value} ({value/total:.2%})")

    def _calculate_metrics(self, 
                         df: pd.DataFrame,
                         evaluate_lemma: bool,
                         evaluate_pos: bool,
                         evaluate_morph: bool) -> pd.DataFrame:
        """Calculate evaluation metrics for each model."""
        metrics = []
        for i, model_name in enumerate(self.model_names):
            model_metrics = [model_name]
            
            if evaluate_lemma:
                gold = df["Gold Lemma"].tolist()
                pred = df[f"{model_name} Lemma"].tolist()
                model_metrics.extend(self._get_metrics(gold, pred))
            
            if evaluate_pos:
                for field in ["POS", "TAG"]:
                    gold = df[f"Gold {field}"].tolist()
                    pred = df[f"{model_name} {field}"].tolist()
                    model_metrics.extend(self._get_metrics(gold, pred))
            
            if evaluate_morph:
                gold = df["Gold Morph"].tolist()
                pred = df[f"{model_name} Morph"].tolist()
                model_metrics.extend(self._get_metrics(gold, pred))
            
            metrics.append(model_metrics)
        
        columns = ["Model"]
        if evaluate_lemma:
            columns.extend(["Lemma Precision", "Lemma Recall", "Lemma F1"])
        if evaluate_pos:
            columns.extend(["POS Precision", "POS Recall", "POS F1"])
            columns.extend(["TAG Precision", "TAG Recall", "TAG F1"])
        if evaluate_morph:
            columns.extend(["Morph Precision", "Morph Recall", "Morph F1"])
        
        return pd.DataFrame(metrics, columns=columns)

    def _get_metrics(self, gold: List[str], pred: List[str]) -> List[float]:
        """Calculate precision, recall, and F1 score."""
        return [
            precision_score(gold, pred, average='weighted', zero_division=0),
            recall_score(gold, pred, average='weighted', zero_division=0),
            f1_score(gold, pred, average='weighted', zero_division=0)
        ]

class NEREvaluator:
    """Evaluates named entity recognition performance across multiple models."""
    
    def __init__(self, models: List[spacy.language.Language], model_names: Optional[List[str]] = None, norm_method: Optional[str] = None):
        """Initialize NER evaluator."""
        self.models = models
        self.model_names = model_names or [f'Model {i+1}' for i in range(len(models))]
        self.norm_method = norm_method
        self.normalizer = TextNormalizer()

    def evaluate_ner(self, docs: List[spacy.tokens.Doc]) -> pd.DataFrame:
        """Evaluate NER performance."""
        data = []
        stats = defaultdict(int)
        gold_labels = []
        predicted_labels = [[] for _ in self.models]
        
        for doc in tqdm(docs, desc="Evaluating NER"):
            doc_text = doc.text
            predicted_docs = [model(doc_text) for model in self.models]
            
            for ent in doc.ents:
                gold_label = ent.label_
                gold_labels.append(gold_label)
                
                # Get predictions from each model
                model_predictions = []
                for i, pred_doc in enumerate(predicted_docs):
                    pred_ent = self._find_matching_entity(ent, pred_doc)
                    pred_label = pred_ent.label_ if pred_ent else None
                    model_predictions.append(pred_label)
                    predicted_labels[i].append(pred_label or 'None')
                
                # Record results
                entry = {
                    "Text": doc_text,
                    "Entity": ent.text,
                    "Gold Label": gold_label,
                    **{f"{name} Label": pred for name, pred in zip(self.model_names, model_predictions)},
                    "Match": all(p == gold_label for p in model_predictions)
                }
                data.append(entry)
                
                if entry["Match"]:
                    stats["matching_predictions"] += 1
                else:
                    stats["different_predictions"] += 1

        df = pd.DataFrame(data)
        self._print_statistics(stats, gold_labels, predicted_labels)
        return df

    def _find_matching_entity(self, 
                            gold_ent: spacy.tokens.Span,
                            doc: spacy.tokens.Doc) -> Optional[spacy.tokens.Span]:
        """Find matching entity in predicted doc."""
        for ent in doc.ents:
            if (abs(ent.start_char - gold_ent.start_char) <= 2 and
                abs(ent.end_char - gold_ent.end_char) <= 2):
                return ent
        return None

    def _print_statistics(self,
                         stats: Dict[str, int],
                         gold_labels: List[str],
                         predicted_labels: List[List[str]]) -> None:
        """Print evaluation statistics."""
        total = stats["matching_predictions"] + stats["different_predictions"]
        logger.info("\nNER Evaluation Results:")
        logger.info(f"Total entities: {total}")
        logger.info(f"Matching predictions: {stats['matching_predictions']} ({stats['matching_predictions']/total:.2%})")
        logger.info(f"Different predictions: {stats['different_predictions']} ({stats['different_predictions']/total:.2%})")
        
        for i, model_name in enumerate(self.model_names):
            precision = precision_score(gold_labels, predicted_labels[i], average='weighted', zero_division=0)
            recall = recall_score(gold_labels, predicted_labels[i], average='weighted', zero_division=0)
            f1 = f1_score(gold_labels, predicted_labels[i], average='weighted', zero_division=0)
            logger.info(f"\n{model_name}:")
            logger.info(f"Precision: {precision:.2%}")
            logger.info(f"Recall: {recall:.2%}")
            logger.info(f"F1 Score: {f1:.2%}")

class SpanCatEvaluator:
    """Evaluates span categorization performance across multiple models."""
    
    def __init__(self, models: List[spacy.language.Language], model_names: Optional[List[str]] = None, norm_method: Optional[str] = None):
        """Initialize span categorizer evaluator."""
        self.models = models
        self.model_names = model_names or [f'Model {i+1}' for i in range(len(models))]
        self.norm_method = norm_method
        self.normalizer = TextNormalizer()

    def evaluate_spans(self, docs: List[spacy.tokens.Doc]) -> pd.DataFrame:
        """Evaluate span categorization performance."""
        data = []
        stats = defaultdict(int)
        gold_labels = []
        predicted_labels = [[] for _ in self.models]
        
        for doc in tqdm(docs, desc="Evaluating spans"):
            doc_text = doc.text
            predicted_docs = [model(doc_text) for model in self.models]
            
            for span in doc.spans["sc"]:
                gold_label = span.label_
                gold_labels.append(gold_label)
                
                # Get predictions from each model
                model_predictions = []
                model_scores = []
                for i, pred_doc in enumerate(predicted_docs):
                    pred_span, score = self._find_matching_span(span, pred_doc)
                    pred_label = pred_span.label_ if pred_span else None
                    model_predictions.append(pred_label)
                    model_scores.append(score)
                    predicted_labels[i].append(pred_label or 'None')
                
                # Record results
                entry = {
                    "Text": doc_text,
                    "Span": span.text,
                    "Gold Label": gold_label,
                    **{f"{name} Label": pred for name, pred in zip(self.model_names, model_predictions)},
                    **{f"{name} Score": score for name, score in zip(self.model_names, model_scores)},
                    "Match": all(p == gold_label for p in model_predictions)
                }
                data.append(entry)
                
                if entry["Match"]:
                    stats["matching_predictions"] += 1
                else:
                    stats["different_predictions"] += 1

        df = pd.DataFrame(data)
        self._print_statistics(stats, gold_labels, predicted_labels)
        return df

    def _find_matching_span(self, 
                          gold_span: spacy.tokens.Span,
                          doc: spacy.tokens.Doc) -> Tuple[Optional[spacy.tokens.Span], float]:
        """Find matching span in predicted doc."""
        best_span = None
        best_score = 0.0
        
        for span in doc.spans["sc"]:
            # Check for overlap
            if (span.start_char <= gold_span.end_char and 
                span.end_char >= gold_span.start_char):
                # Calculate overlap score
                overlap = min(span.end_char, gold_span.end_char) - max(span.start_char, gold_span.start_char)
                union = max(span.end_char, gold_span.end_char) - min(span.start_char, gold_span.start_char)
                score = overlap / union if union > 0 else 0
                
                if score > best_score:
                    best_span = span
                    best_score = score
        
        return best_span, best_score

    def _print_statistics(self,
                         stats: Dict[str, int],
                         gold_labels: List[str],
                         predicted_labels: List[List[str]]) -> None:
        """Print evaluation statistics."""
        total = stats["matching_predictions"] + stats["different_predictions"]
        logger.info("\nSpan Categorization Results:")
        logger.info(f"Total spans: {total}")
        logger.info(f"Matching predictions: {stats['matching_predictions']} ({stats['matching_predictions']/total:.2%})")
        logger.info(f"Different predictions: {stats['different_predictions']} ({stats['different_predictions']/total:.2%})")
        
        for i, model_name in enumerate(self.model_names):
            precision = precision_score(gold_labels, predicted_labels[i], average='weighted', zero_division=0)
            recall = recall_score(gold_labels, predicted_labels[i], average='weighted', zero_division=0)
            f1 = f1_score(gold_labels, predicted_labels[i], average='weighted', zero_division=0)
            logger.info(f"\n{model_name}:")
            logger.info(f"Precision: {precision:.2%}")
            logger.info(f"Recall: {recall:.2%}")
            logger.info(f"F1 Score: {f1:.2%}")