#!/usr/bin/env python3
"""
Utility functions for building and managing lemmatization dictionaries.
"""

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict

import pandas as pd
from cassis import load_typesystem, load_cas_from_xmi

from .utils import TextNormalizer, NormalizationForm, PUNCTUATION

logger = logging.getLogger(__name__)

# Source priority weights - higher values indicate more reliable sources
SOURCE_WEIGHTS = {
    'Inception': 4.0,  # Most reliable - manually annotated
    'Coda': 3.0,      # Manually curated dataset
    'Conllu': 2.0,    # Universal Dependencies format
    'Dendrosearch': 1.0  # Automated extraction
}

class DictionaryBuilder:
    """Handles creation and management of lemmatization dictionaries."""
    
    @staticmethod
    def build_coda_dict(df: pd.DataFrame) -> Dict[str, str]:
        """
        Create dictionary from CODA dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing CODA data with 'Keyword' and 'Lemma' columns
            
        Returns:
            Dict[str, str]: Dictionary mapping keywords to lemmas
        """
        return df.dropna(subset=['Keyword', 'Lemma']).set_index('Keyword')['Lemma'].to_dict()

    @staticmethod
    def build_dendrosearch_dict(file_path: str) -> Dict[str, str]:
        """
        Create dictionary from Dendrosearch file.
        
        Args:
            file_path (str): Path to Dendrosearch file
            
        Returns:
            Dict[str, str]: Dictionary mapping words to lemmas
        """
        normalizer = TextNormalizer()
        dendrosearch_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dendrosearch_dict = {
                line.split()[0]: line.split()[1] 
                for line in f 
                if len(line.split()) > 1 and line.split()[0] not in PUNCTUATION
            }
            
        return {
            normalizer.normalize_text(k, standardize_apostrophe=True): 
            normalizer.normalize_text(v, lowercase=True, standardize_apostrophe=True)
            for k, v in dendrosearch_dict.items()
        }

    @staticmethod
    def build_conllu_dict(input_path: str, debug: bool = False) -> Dict[str, str]:
        """
        Create dictionary from CONLLU files.
        
        Args:
            input_path (str): Path to directory containing CONLLU files
            debug (bool): Enable debug logging
            
        Returns:
            Dict[str, str]: Dictionary mapping words to lemmas
        """
        conllu_dict = {}
        
        def process_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if not line.strip() or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) > 2 and parts[1] not in PUNCTUATION:
                        conllu_dict[parts[1]] = parts[2]
            
        for file_path in Path(input_path).glob("*.conllu"):
            if debug:
                logger.debug(f"Processing CONLLU file: {file_path.name}")
            process_file(file_path)
            
        return conllu_dict

    @staticmethod
    def process_inception_files(inception_files_path: Path) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
        """
        Process INCEpTION files and extract lemma dictionaries and sentences.
        
        Args:
            inception_files_path (Path): Path to directory containing INCEpTION zip files
            
        Returns:
            Tuple[Dict[str, str], List[Tuple[str, str]]]: Dictionary of lemmas and list of sentence-source pairs
        """
        inception_dict = {}  # Dictionary to store lemma mappings
        inception_sentences = []
        
        # Create a temporary directory for extraction
        tempdir_path = Path(tempfile.mkdtemp())
        
        try:
            # Process each zip file
            for zip_file_path in inception_files_path.glob("*.zip"):
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(tempdir_path)
                    logger.info(f"Extracting {zip_file_path.name}")
                    
                # Find and load TypeSystem.xml
                typesystem_file_path = tempdir_path / "TypeSystem.xml"
                if not typesystem_file_path.exists():
                    logger.warning(f"No TypeSystem.xml found in {zip_file_path}")
                    continue
                    
                # Load the typesystem
                with open(typesystem_file_path, 'rb') as f:
                    typesystem = load_typesystem(f)
                    
                # Process each XMI file
                for xmi_file_path in tempdir_path.glob("*.xmi"):
                    try:
                        with open(xmi_file_path, 'rb') as f:
                            logger.info(f"Processing file: {xmi_file_path.name}")
                            
                            # Load CAS with typesystem
                            cas = load_cas_from_xmi(f, typesystem=typesystem, lenient=True)
                            
                            # Update inception_dict with lemmas
                            inception_dict.update({
                                token.get_covered_text(): token.value
                                for token in cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma')
                            })
                            
                            # Extend inception_sentences with sentences and their source
                            inception_sentences.extend([
                                (
                                    TextNormalizer.normalize_text(
                                        ' '.join(sentence.get_covered_text().replace('\r', ' ').replace('\n', ' ').split()),
                                        standardize_apostrophe=True
                                    ),
                                    xmi_file_path.name
                                )
                                for sentence in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
                            ])
                            
                    except Exception as e:
                        logger.error(f"Error processing {xmi_file_path.name}: {e}")
                        
        finally:
            # Print how many sentences were processed
            logger.info(f"Processed {len(inception_sentences)} sentences.")
            DictionaryBuilder.cleanup_tempdir(tempdir_path)
            
        return inception_dict, inception_sentences

    @staticmethod
    def cleanup_tempdir(directory: Path):
        """
        Helper method to cleanup temporary directory and its contents.
        
        Args:
            directory (Path): Path to the temporary directory to clean up
        """
        for child in directory.iterdir():
            if child.is_file():
                child.unlink()
        directory.rmdir()

    @staticmethod
    def combine_dictionaries(dictionaries: Dict[str, Dict[str, str]], 
                           normalization_forms: List[str] = ['NFC'],
                           debug: bool = False) -> Dict[NormalizationForm, Dict[str, Dict[str, Set[str]]]]:
        """
        Combine multiple dictionaries with source tracking and normalization.
        
        Args:
            dictionaries: Dictionary of dictionaries with source names as keys
            normalization_forms: List of Unicode normalization forms to use
            debug: Enable debug logging
            
        Returns:
            Dictionary of processed pairs with sources for each normalization form
        """
        processed_pairs = {NormalizationForm.from_string(form): {} for form in normalization_forms}
        
        for form in normalization_forms:
            for source, lemma_dict in dictionaries.items():
                for word, lemma in lemma_dict.items():
                    # Normalize word and lemma
                    norm_word = TextNormalizer.normalize_text(
                        word, form=form, 
                        remove_accents=False, 
                        lowercase=False, 
                        standardize_apostrophe=True, 
                        remove_brackets=True, 
                        remove_trailing_numbers=True, 
                        debug=debug
                    )
                    
                    norm_lemma = TextNormalizer.normalize_text(
                        lemma, form=form,
                        remove_accents=False,
                        lowercase=True,
                        standardize_apostrophe=True,
                        remove_brackets=True,
                        remove_trailing_numbers=True,
                        debug=debug
                    )

                    # Skip empty entries
                    if not norm_word or norm_lemma in ["_", " ", ""]:
                        continue

                    # Initialize word entry if needed
                    if norm_word not in processed_pairs[form]:
                        processed_pairs[form][norm_word] = {}

                    # Add or update lemma with source
                    if norm_lemma in processed_pairs[form][norm_word]:
                        processed_pairs[form][norm_word][norm_lemma].add(source)
                    else:
                        processed_pairs[form][norm_word][norm_lemma] = {source}

        return processed_pairs

    @staticmethod
    def calculate_lemma_score(sources: Set[str], frequency: int) -> float:
        """
        Calculate a score for a lemma based on source weights and frequency.
        
        Args:
            sources: Set of source names that provided this lemma
            frequency: Number of times this lemma appears
            
        Returns:
            Float score representing the reliability of this lemma
        """
        # Base score from frequency
        score = float(frequency)
        
        # Add weighted source scores
        source_score = sum(SOURCE_WEIGHTS.get(source, 1.0) for source in sources)
        
        # Combine frequency and source scores
        # We multiply by source score to give more weight to reliable sources
        final_score = score * source_score
        
        return final_score

    @staticmethod
    def process_pairs(processed_pairs: Dict[NormalizationForm, Dict[str, Dict[str, Set[str]]]], 
                     debug: bool = False) -> Dict[NormalizationForm, Dict[str, Dict[str, Set[str]]]]:
        """
        Process pairs and choose the most common lemma for each word.
        
        Args:
            processed_pairs: Dictionary of processed word-lemma pairs with sources
            debug: Enable debug logging
            
        Returns:
            Dictionary of processed pairs with only the most common lemma for each word
        """
        processed_counter = 0  # Counter for processed words
        no_lemma_counter = 0  # Counter for words with no lemma
        deleted_counter = 0  # Counter for deleted words
        
        for form in processed_pairs:
            keys_to_delete = []  # List to store keys of pairs to be deleted
            
            for word, lemmas in processed_pairs[form].items():
                best_lemma = None
                best_score = 0.0
                best_sources = set()
                tied_lemmas = {}  # Dictionary to store lemmas with their scores
                
                for lemma, sources in lemmas.items():
                    current_count = len(sources)
                    current_score = DictionaryBuilder.calculate_lemma_score(sources, current_count)
                    
                    if current_score > best_score:
                        best_lemma = lemma
                        best_score = current_score
                        best_sources = sources
                        tied_lemmas = {lemma: (current_score, sources)}
                    elif abs(current_score - best_score) < 1e-10:  # Handle floating point comparison
                        tied_lemmas[lemma] = (current_score, sources)
                
                # Handle ties: choose the lexically first lemma
                if len(tied_lemmas) > 1:  
                    # Sort by lemma text for consistent results
                    sorted_lemmas = sorted(tied_lemmas.items())
                    best_lemma = sorted_lemmas[0][0]  # Take the first lemma
                    best_score, best_sources = sorted_lemmas[0][1]
                    # Combine sources of tied lemmas since more than one had the "most common" status
                    best_sources = set().union(*(sources for _, (_, sources) in sorted_lemmas))
                
                if best_lemma:
                    processed_pairs[form][word] = {best_lemma: best_sources}
                    processed_counter += 1
                else:
                    keys_to_delete.append(word)
                    no_lemma_counter += 1
            
            # Delete marked words outside the loop
            for word in keys_to_delete:
                del processed_pairs[form][word]
                deleted_counter += 1
        
        if debug:
            logger.info(f"Total processed words: {processed_counter}\nTotal words with no lemma: {no_lemma_counter}\nTotal deleted words: {deleted_counter}")
        return processed_pairs