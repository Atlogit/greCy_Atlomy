#!/usr/bin/env python3
"""
Span categorization preprocessing module.
"""

import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict

import spacy
import pandas as pd
from spacy.tokens import Doc, Span, SpanGroup
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import TextNormalizer, NormalizationForm
from .file_io import FileIO
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)

class Config:
    """Default configuration settings."""
    DEFAULT_FORM = 'NFKD'
    REMOVE_ACCENTS = False
    LOWERCASE = False
    STANDARDIZE_APOSTROPHE = True
    REMOVE_BRACKETS = False
    REMOVE_TRAILING_NUMBERS = False
    REMOVE_EXTRA_SPACES = False
    DEBUG = False

class SpanProcessor:
    """Handles span annotation processing."""
    
    def __init__(self, nlp: spacy.language.Language):
        """
        Initialize span processor.
        
        Args:
            nlp: Loaded spaCy model
        """
        self.nlp = nlp
        self.normalizer = TextNormalizer()
        self.config = Config()
        
        # Ensure spans extension is set
        if not Span.has_extension("labels"):
            Span.set_extension("labels", default=[], force=True)

    def process_sentence(self, 
                        text: str,
                        entities: List[Tuple[int, int, str]],
                        source_info: str,
                        form: NormalizationForm = NormalizationForm.NFC,
                        debug: bool = False) -> Optional[Doc]:
        """
        Process a sentence with its span annotations.
        
        Args:
            text (str): The sentence text
            entities (List[Tuple[int, int, str]]): List of entity tuples (start, end, label)
            source_info (str): Source information for the sentence
            form (NormalizationForm): Unicode normalization form
            debug (bool): Enable debug logging
            
        Returns:
            Optional[Doc]: Processed spaCy Doc or None if processing fails
        """
        try:
            # Normalize text
            normalized_text = self.normalizer.normalize_text(
                text,
                form=form,
                remove_accents=self.config.REMOVE_ACCENTS,
                lowercase=self.config.LOWERCASE,
                standardize_apostrophe=self.config.STANDARDIZE_APOSTROPHE,
                remove_brackets=self.config.REMOVE_BRACKETS,
                remove_trailing_numbers=self.config.REMOVE_TRAILING_NUMBERS,
                remove_extra_spaces=self.config.REMOVE_EXTRA_SPACES
            )
            
            # Create base doc
            doc = self.nlp.make_doc(normalized_text)
            spans = []
            
            # Process each entity
            for start, end, label in entities:
                if isinstance(label, float) and np.isnan(label):
                    logger.warning(f"Skipping entity with 'nan' label: {text[start:end]} [{start},{end}]")
                    continue
                
                # Convert single label to list
                labels = [label] if isinstance(label, str) else label
                
                # Validate labels
                if not all(l == l and l is not None for l in labels):
                    continue
                
                # Find token span
                tokens = [token for token in doc if not (token.idx + len(token.text) <= start or token.idx >= end)]
                if not tokens:
                    continue
                
                # Adjust span boundaries to token boundaries
                span_start = tokens[0].idx
                span_end = tokens[-1].idx + len(tokens[-1].text)
                
                # Create spans for each label
                for label in labels:
                    span = doc.char_span(span_start, span_end, label=label, alignment_mode="expand")
                    if span is not None:
                        spans.append(span)
                    else:
                        logger.warning(f"Invalid span: {text[start:end]} [{start},{end}]")
            
            # Add spans to doc
            doc.spans["sc"] = spans
            doc.user_data["source_info"] = source_info
            
            if debug:
                logger.debug(f"Processed sentence with {len(spans)} spans")
                logger.debug(f"Text: {doc.text}")
                logger.debug(f"Spans: {[(span.text, span.label_) for span in spans]}")
            
            return doc
            
        except Exception as e:
            logger.error(f"Error processing sentence: {e}")
            if debug:
                logger.debug("Stack trace:", exc_info=True)
            return None

    def process_data(self, 
                    data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]], str]],
                    form: NormalizationForm = NormalizationForm.NFC,
                    debug: bool = False) -> List[Doc]:
        """
        Process a list of annotated sentences.
        
        Args:
            data: List of (text, annotations, source) tuples
            form: Unicode normalization form
            debug: Enable debug logging
            
        Returns:
            List[Doc]: List of processed spaCy Docs
        """
        docs = []
        missing_spans = []
        total_spans = 0
        processed_docs = 0
        
        for text, annot, source_info in tqdm(data, desc="Processing sentences"):
            doc = self.process_sentence(text, annot['entities'], source_info, form=form, debug=debug)
            if doc is not None:
                docs.append(doc)
                processed_docs += 1
                total_spans += len(doc.spans["sc"])
        
        logger.info(f"Processed {processed_docs} documents with {total_spans} total spans")
        if missing_spans:
            logger.warning(f"Failed to process {len(missing_spans)} spans")
        
        return docs

    def split_data(self,
                   docs: List[Doc],
                   output_dir: Path,
                   train_size: float = 0.8,
                   random_state: int = 42,
                   form: str = "NFC",
                   debug: bool = False) -> None:
        """
        Split and save the processed documents.
        
        Args:
            docs (List[Doc]): Documents to split
            output_dir (Path): Output directory
            train_size (float): Proportion of data for training
            random_state (int): Random seed
            form (str): Normalization form for file naming
            debug (bool): Enable debug logging
        """
        if not docs:
            logger.warning("No documents to split")
            return
            
        # Create label array for stratification
        labels_array = self._create_label_array(docs)
        
        # Split into train and test
        train_docs, test_docs = train_test_split(
            docs,
            train_size=train_size,
            random_state=random_state,
            stratify=labels_array
        )
        
        # Split test into dev and test
        dev_docs, test_docs = train_test_split(
            test_docs,
            test_size=0.5,
            random_state=random_state,
            stratify=self._create_label_array(test_docs)
        )
        
        # Save splits
        splits = {
            'train': train_docs,
            'dev': dev_docs,
            'test': test_docs
        }
        
        for split_name, split_docs in splits.items():
            output_path = output_dir / split_name / "spancat" / f"spancat_{split_name}_{form}.spacy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            FileIO.save_docs(split_docs, output_path)
            
            if debug:
                logger.debug(f"Saved {len(split_docs)} documents to {output_path}")
                
        logger.info(f"Split sizes - Train: {len(train_docs)}, Dev: {len(dev_docs)}, Test: {len(test_docs)}")

    def _create_label_array(self, docs: List[Doc]) -> np.ndarray:
        """Create a binary label array for stratification."""
        all_labels = set()
        for doc in docs:
            for span in doc.spans["sc"]:
                all_labels.add(span.label_)
        
        unique_labels = sorted(all_labels)
        labels_array = np.zeros((len(docs), len(unique_labels)), dtype=int)
        
        for i, doc in enumerate(docs):
            doc_labels = set(span.label_ for span in doc.spans["sc"])
            labels_array[i, [unique_labels.index(label) for label in doc_labels]] = 1
        
        return labels_array

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Span Categorization Preprocessing")
    parser.add_argument("--input-dir", required=True, help="Input directory containing span files")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--model", default="grc_proiel_trf", help="SpaCy model to use")
    parser.add_argument("--form", default="NFC", choices=[f.name for f in NormalizationForm],
                       help="Normalization form to use")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Load spaCy model
        nlp = spacy.load(args.model)
        
        # Add spancat if not present
        if "spancat" not in nlp.pipe_names:
            spancat = nlp.add_pipe("spancat", last=True)
            spancat.cfg["spans_key"] = "sc"
        
        # Initialize processor
        processor = SpanProcessor(nlp)
        
        # Load and process data
        # This would need to be adapted based on your input data format
        logger.info("Loading and processing data...")
        
        # Split and save
        processor.split_data(
            docs,  # Your processed docs
            Path(args.output_dir),
            form=args.form,
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()