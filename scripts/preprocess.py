#!/usr/bin/env python3
"""
Common preprocessing routines for text and data preparation.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

import spacy
from spacy.tokens import Doc

from .utils import TextNormalizer, NormalizationForm
from .file_io import FileIO

logger = logging.getLogger(__name__)

class Preprocessor:
    """Handles common preprocessing operations for text data."""
    
    def __init__(self, nlp: Optional[Any] = None):
        """
        Initialize preprocessor with optional spaCy model.
        
        Args:
            nlp: Optional loaded spaCy model
        """
        self.nlp = nlp
        self.normalizer = TextNormalizer()

    def load_spacy_model(self, model_name: str, disable: List[str] = None) -> None:
        """
        Load a spaCy model with specified components disabled.
        
        Args:
            model_name (str): Name of the spaCy model to load
            disable (List[str]): List of pipeline components to disable
        """
        try:
            self.nlp = spacy.load(model_name)
            if disable:
                self.nlp.disable_pipes(*disable)
            logger.info(f"Loaded spaCy model '{model_name}'")
        except Exception as e:
            logger.error(f"Error loading spaCy model '{model_name}': {e}")
            raise

    def process_text(self, text: str,
                    form: NormalizationForm = NormalizationForm.NFC,
                    remove_accents: bool = False,
                    lowercase: bool = False,
                    standardize_apostrophe: bool = True,
                    remove_brackets: bool = False,
                    remove_trailing_numbers: bool = False,
                    remove_extra_spaces: bool = True) -> str:
        """
        Process text using configured normalization settings.
        
        Args:
            text (str): Input text to process
            form (NormalizationForm): Unicode normalization form
            remove_accents (bool): Whether to remove diacritical marks
            lowercase (bool): Whether to convert to lowercase
            standardize_apostrophe (bool): Whether to standardize apostrophes
            remove_brackets (bool): Whether to remove brackets
            remove_trailing_numbers (bool): Whether to remove numbers at start/end
            remove_extra_spaces (bool): Whether to remove extra whitespace
            
        Returns:
            str: Processed text
        """
        return self.normalizer.normalize_text(
            text,
            form=form,
            remove_accents=remove_accents,
            lowercase=lowercase,
            standardize_apostrophe=standardize_apostrophe,
            remove_brackets=remove_brackets,
            remove_trailing_numbers=remove_trailing_numbers,
            remove_extra_spaces=remove_extra_spaces
        )

    def process_batch(self, texts: List[str], batch_size: int = 1000, **kwargs) -> List[str]:
        """
        Process a batch of texts with the same settings.
        
        Args:
            texts (List[str]): List of texts to process
            batch_size (int): Size of batches for processing
            **kwargs: Arguments to pass to process_text
            
        Returns:
            List[str]: List of processed texts
        """
        processed_texts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            processed_texts.extend([
                self.process_text(text, **kwargs)
                for text in batch
            ])
        return processed_texts

    def create_docs(self, texts: List[str], batch_size: int = 1000) -> List[Doc]:
        """
        Create spaCy Doc objects from texts.
        
        Args:
            texts (List[str]): List of texts to process
            batch_size (int): Size of batches for processing
            
        Returns:
            List[Doc]: List of spaCy Doc objects
            
        Raises:
            ValueError: If no spaCy model is loaded
        """
        if not self.nlp:
            raise ValueError("No spaCy model loaded. Call load_spacy_model first.")
            
        docs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            docs.extend(list(self.nlp.pipe(batch)))
        return docs

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Apply basic text cleaning operations.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[\u2018\u2019\']', "'", text)  # Using unicode escapes for quotes
        
        return text.strip()

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using basic rules.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Basic sentence splitting on punctuation followed by space and uppercase
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def prepare_training_data(self, 
                            input_texts: List[str],
                            output_dir: Path,
                            train_split: float = 0.8,
                            dev_split: float = 0.1,
                            seed: int = 42,
                            **kwargs) -> Tuple[List[Doc], List[Doc], List[Doc]]:
        """
        Prepare and split data for training.
        
        Args:
            input_texts (List[str]): List of input texts
            output_dir (Path): Directory to save processed files
            train_split (float): Proportion of data for training
            dev_split (float): Proportion of data for development
            seed (int): Random seed for reproducibility
            **kwargs: Additional arguments for text processing
            
        Returns:
            Tuple[List[Doc], List[Doc], List[Doc]]: Train, dev, and test docs
        """
        import random
        random.seed(seed)
        
        # Process texts
        processed_texts = self.process_batch(input_texts, **kwargs)
        
        # Create docs
        docs = self.create_docs(processed_texts)
        random.shuffle(docs)
        
        # Split data
        n = len(docs)
        train_size = int(n * train_split)
        dev_size = int(n * dev_split)
        
        train_docs = docs[:train_size]
        dev_docs = docs[train_size:train_size + dev_size]
        test_docs = docs[train_size + dev_size:]
        
        # Save splits
        FileIO.ensure_dir(output_dir)
        FileIO.save_docs(train_docs, output_dir / "train.spacy")
        FileIO.save_docs(dev_docs, output_dir / "dev.spacy")
        FileIO.save_docs(test_docs, output_dir / "test.spacy")
        
        return train_docs, dev_docs, test_docs