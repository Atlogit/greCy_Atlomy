"""
Greek Text Preprocessing Pipeline

A modular preprocessing pipeline for Greek text data, supporting:
- Lemmatization
- Morphology/POS tagging
- Span categorization
- Document analysis and comparison

See README.md for usage instructions.
"""

from .utils import TextNormalizer, NormalizationForm
from .dictionary_utils import DictionaryBuilder, SOURCE_WEIGHTS
from .file_io import FileIO
from .preprocess import Preprocessor
from .lemma_preprocess import SentenceProcessor
from .morphology_preprocess import MorphologyProcessor
from .spancat_preprocess import SpanProcessor
from .conllu_utils import ConlluProcessor
from .analysis_utils import DocumentAnalyzer

__version__ = "0.1.0"
__all__ = [
    # Core utilities
    "TextNormalizer",
    "NormalizationForm",
    "DictionaryBuilder",
    "SOURCE_WEIGHTS",
    "FileIO",
    "Preprocessor",
    
    # Processing modules
    "SentenceProcessor",
    "MorphologyProcessor",
    "SpanProcessor",
    "ConlluProcessor",
    
    # Analysis utilities
    "DocumentAnalyzer",
]

# Configure default logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)