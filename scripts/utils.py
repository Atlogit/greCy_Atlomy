#!/usr/bin/env python3
"""
Core utility functions and classes for text processing and logging.
"""

import logging
import unicodedata
import re
from enum import Enum
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NormalizationForm(Enum):
    """Supported Unicode normalization forms."""
    NFC = 'NFC'
    NFKC = 'NFKC'
    NFD = 'NFD'
    NFKD = 'NFKD'

    @classmethod
    def from_string(cls, form: str) -> 'NormalizationForm':
        """Convert string to NormalizationForm enum."""
        try:
            return cls[form.upper()]
        except KeyError:
            valid_forms = [f.name for f in cls]
            raise ValueError(
                f"Invalid normalization form: {form}. Must be one of {valid_forms}")

# Constants
APOSTROPHES = ["᾽", "᾿", "'", "'", "'"]
CORRECT_APOSTROPHE = "ʼ"
PUNCTUATION = set(['.', ")", "·", "(", "[", "]", ":", ";", ",", "?", "!", "،", "_"])
ALLOWED_CHARACTERS = [' ̓', "᾿", "᾽", "'", "'", "'", 'ʼ', '̓']

class TextNormalizer:
    """Handles text normalization and cleaning operations."""
    
    @staticmethod
    def clean_and_remove_accents(text: str) -> str:
        """
        Clean text by removing diacritics except for specific characters.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text with diacritics removed except for allowed characters
            
        Raises:
            ValueError: If input is not a string
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
            
        try:
            non_accent_chars = [
                c for c in unicodedata.normalize('NFC', text)
                if unicodedata.category(c) != 'Mn' or c in ALLOWED_CHARACTERS
            ]
            return ''.join(non_accent_chars)
        except Exception as e:
            logger.error(f"Error in clean_and_remove_accents: {e}")
            return text

    @staticmethod
    def normalize_text(text: str, 
                      form: Union[str, NormalizationForm] = NormalizationForm.NFC,
                      remove_accents: bool = False,
                      lowercase: bool = False,
                      standardize_apostrophe: bool = True,
                      remove_brackets: bool = False,
                      remove_trailing_numbers: bool = False,
                      remove_extra_spaces: bool = False,
                      debug: bool = False) -> str:
        """
        Apply multiple text normalization steps.
        
        Args:
            text (str): Input text to normalize
            form (Union[str, NormalizationForm]): Unicode normalization form to use
            remove_accents (bool): Whether to remove diacritical marks
            lowercase (bool): Whether to convert text to lowercase
            standardize_apostrophe (bool): Whether to standardize apostrophe characters
            remove_brackets (bool): Whether to remove brackets
            remove_trailing_numbers (bool): Whether to remove numbers at start/end
            remove_extra_spaces (bool): Whether to remove extra whitespace
            debug (bool): Whether to print debug information
            
        Returns:
            str: Normalized text
        """
        if not isinstance(text, str):
            return ""
            
        if isinstance(form, str):
            form = NormalizationForm.from_string(form)
            
        normalized_text = text

        def debug_print(operation_name, before, after):
            if debug:
                logger.debug(f"{operation_name} - Before: {before}")
                logger.debug(f"{operation_name} - After: {after}")

        # Standardize apostrophes
        if standardize_apostrophe:
            before_text = normalized_text
            for apos in APOSTROPHES:
                normalized_text = normalized_text.replace(apos, CORRECT_APOSTROPHE)
            debug_print("Standardizing apostrophes", before_text, normalized_text)

        # Remove accents
        if remove_accents:
            before_text = normalized_text
            try:
                normalized_text = TextNormalizer.clean_and_remove_accents(normalized_text)
            except Exception as e:
                logger.error(f"Error removing accents: {e}")
                return text
            debug_print("Removing accents", before_text, normalized_text)

        # Convert to lowercase
        if lowercase:
            before_text = normalized_text
            normalized_text = normalized_text.lower()
            debug_print("Lowercase conversion", before_text, normalized_text)

        # Unicode normalization
        if form:
            before_text = normalized_text
            normalized_text = unicodedata.normalize(form.value, normalized_text)
            debug_print("Unicode normalization", before_text, normalized_text)

        # Remove brackets
        if remove_brackets:
            before_text = normalized_text
            normalized_text = re.sub(r'[\(\)\[\]]', '', normalized_text)
            debug_print("Removing brackets", before_text, normalized_text)

        # Remove trailing numbers
        if remove_trailing_numbers:
            before_text = normalized_text
            normalized_text = re.sub(r'^\d+|\d+$', '', normalized_text)
            debug_print("Removing trailing numbers", before_text, normalized_text)

        # Remove extra spaces
        if remove_extra_spaces:
            before_text = normalized_text
            normalized_text = ' '.join(normalized_text.split()).strip()
            debug_print("Removing extra spaces", before_text, normalized_text)

        return normalized_text