#!/usr/bin/env python3
"""
Greek Lemmatizer Script
Processes Greek text using various sources for lemmatization and saves the results.
"""

import os
import re
import json
import random
import logging
from enum import Enum
import argparse
import unicodedata
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import OrderedDict, defaultdict

import spacy
import pandas as pd
from tqdm import tqdm
from spacy.tokens import Doc, DocBin
from cassis import load_typesystem, load_cas_from_xmi

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
APOSTROPHES = ["᾽", "᾿", "'", "’", "‘"]
CORRECT_APOSTROPHE = "ʼ"
PUNCTUATION = set(['.', ")", "·", "(", "[", "]", ":", ";", ",", "?", "!", "،", "_"])
ALLOWED_CHARACTERS = [' ̓', "᾿", "᾽", "'", "’", "‘", 'ʼ', '̓']

# Source priority weights - higher values indicate more reliable sources
SOURCE_WEIGHTS = {
    'Inception': 4.0,  # Most reliable - manually annotated
    'Coda': 3.0,      # Manually curated dataset
    'Conllu': 2.0,    # Universal Dependencies format
    'Dendrosearch': 1.0  # Automated extraction
}

class TextNormalizer:
    """Handles text normalization and cleaning operations."""
    
    @staticmethod
    def clean_and_remove_accents(text: str) -> str:
        """Clean text by removing diacritics except for specific characters."""
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
        """Apply multiple text normalization steps."""
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

class DictionaryBuilder:
    """Handles creation and management of lemmatization dictionaries."""
    
    @staticmethod
    def build_coda_dict(df: pd.DataFrame) -> Dict[str, str]:
        """Create dictionary from CODA dataset."""
        return df.dropna(subset=['Keyword', 'Lemma']).set_index('Keyword')['Lemma'].to_dict()

    @staticmethod
    def build_dendrosearch_dict(file_path: str) -> Dict[str, str]:
        """Create dictionary from Dendrosearch file."""
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
        """Create dictionary from CONLLU files."""
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
        """Process INCEpTION files and extract lemma dictionaries and sentences."""
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
                        
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            DictionaryBuilder.cleanup_tempdir(tempdir_path)
            raise  # Re-raise to allow caller to handle
            
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            DictionaryBuilder.cleanup_tempdir(tempdir_path)
            raise  # Re-raise to allow caller to handle
            
        else:
            # This block runs if no exceptions were raised
            logger.info("\nFinito!")
            
        finally:
            # Print how many sentences were processed
            logger.info(f"Processed {len(inception_sentences)} sentences.")
            DictionaryBuilder.cleanup_tempdir(tempdir_path)
            
        return inception_dict, inception_sentences

    def cleanup_tempdir(directory: Path):
        """
        Helper method to cleanup temporary directory and its contents.
        
        Args:
            directory (Path): Path to the temporary directory to clean up.
            
        Note: This is used internally by process_inception_files."""
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

class SentenceProcessor:
    """Handles sentence processing and lemmatization."""
    
    @staticmethod
    def process_sentences(sentences_list: List[Tuple[str, str]], 
                        nlp: Any,
                        processed_pairs: Dict[NormalizationForm, Dict[str, Dict[str, Set[str]]]],
                        debug: bool = False,
                        batch_size: int = 1000) -> Tuple[List[Dict], List[Doc]]:
        """
        Process sentences using spaCy and apply lemmatization corrections.
        
        Args:
            sentences_list: List of (sentence, source) tuples
            nlp: Loaded spaCy model
            processed_pairs: Dictionary of processed word-lemma pairs with sources
            debug: Enable debug logging
            batch_size: Number of sentences to process in each batch
            
        Returns:
            Tuple of (corrections list, processed docs list)
        """
        corrections = []
        processed_docs = []
        
        # Prepare sentences with metadata
        sentences_and_metadata = [
            (
                TextNormalizer.normalize_text(
                    sentence,
                    form=NormalizationForm.NFC,
                    remove_accents=False,
                    lowercase=False,
                    standardize_apostrophe=True,
                    remove_brackets=False,
                    remove_trailing_numbers=False,
                    debug=debug
                ),
                (source_info, NormalizationForm.NFC)
            )
            for sentence, source_info in sentences_list
        ]
        
        sentences_for_processing = [sentence for sentence, _ in sentences_and_metadata]
        metadata = [meta for _, meta in sentences_and_metadata]
        
        corrected_count = 0
        not_corrected_count = 0
        
        # Process sentences
        for doc, meta in tqdm(zip(nlp.pipe(sentences_for_processing, batch_size=batch_size), metadata),
                            total=len(sentences_and_metadata)):
            form = meta[1] if isinstance(meta[1], NormalizationForm) else NormalizationForm.from_string(meta[1])
            
            if debug:
                logger.debug(f"Processing sentence: {doc.text}")
                logger.debug(f"Form: {form}")
            
            doc.user_data["source_info"] = meta[0]
            
            for token in doc:
                lemma_sources = None
                
                if token.text in processed_pairs[form]:
                    lemma_sources = processed_pairs[form][token.text]
                elif token.text.lower() in processed_pairs[form]:
                    lemma_sources = processed_pairs[form][token.text.lower()]
                    
                if lemma_sources is not None:
                    for lemma, sources in lemma_sources.items():
                        if lemma != token.lemma_:
                            corrections.append({
                                'sentence': doc.text,
                                'source_info': meta[0],
                                'token': token.text,
                                'lemma': token.lemma_,
                                'lemma_corrected': lemma,
                                'correction_source': ', '.join(sources)
                            })
                            token.lemma_ = lemma
                            corrected_count += 1
                            
                            if debug:
                                logger.debug(f"Corrected token '{token.text}' to '{lemma}'")
                            break
                    else:
                        not_corrected_count += 1
                else:
                    not_corrected_count += 1
                    token.lemma_ = ""
            
            processed_docs.append(doc)
            
        if debug:
            logger.debug(f"Corrected {corrected_count} tokens")
            logger.debug(f"Did not correct {not_corrected_count} tokens")
            
        return corrections, processed_docs

class DocumentProcessor:
    """Handles processing and saving of spaCy documents."""
    
    @staticmethod
    def modify_token_attributes(doc: Doc, debug: bool = False) -> Doc:
        """Modify token attributes according to specific rules."""
        counts = defaultdict(int)
        
        for token in doc:
            # Set empty lemma for trainable lemmatizer
            if token.lemma_ in ['', "_", '—', '-']:
                token.lemma_ = ''
                counts['lemma'] += 1
                
            # Set empty POS tags
            if token.pos_ in ['', "_", '—', '-', 'X', 'END', 'MID']:
                token.pos_ = ""
                counts['pos'] += 1
                
            # Set dependency labels
            if token.dep_ in ['', "_", '—', '-']:
                token.dep_ = "None"
                counts['dep'] += 1
                
            if token.head.dep_ in ['', "_", '—', '-']:
                token.head.dep_ = "None"
                counts['head_dep'] += 1
                
        if debug:
            for key, count in counts.items():
                logger.debug(f"Total {key} adjusted: {count}")
                
        return doc

    @staticmethod
    def save_docs(docs: List[Doc], output_path: Path, split_name: str, form: NormalizationForm = NormalizationForm.NFC):
        """Save processed documents to disk."""
        doc_bin = DocBin(docs=docs, store_user_data=True)
        output_file = output_path / f"{split_name}.spacy"
        doc_bin.to_disk(output_file)
        logger.info(f"Saved {len(docs)} documents to {output_file}")

    @staticmethod
    def split_and_save_docs(docs: List[Doc], 
                           base_path: Path,
                           forms: List[str] = ["NFC"],
                           train_size: float = 0.8, 
                           random_seed: int = 42):
        """Split documents into train/dev/test sets and save them."""
        random.seed(random_seed)
        
        # Create output directories
        for split in ['train', 'dev', 'test']:
            (base_path / split / 'lemma_{split}').mkdir(parents=True, exist_ok=True)
        
        # Split documents
        train_size = int(len(docs) * train_size)
        temp_size = len(docs) - train_size
        dev_size = temp_size // 2
        
        indices = list(range(len(docs)))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        dev_indices = indices[train_size:train_size + dev_size]
        test_indices = indices[train_size + dev_size:]
        
        # Save splits for each normalization form
        for form_str in forms:
            form = NormalizationForm.from_string(form_str)
            form_suffix = f"_{form.name}"
            
            DocumentProcessor.save_docs(
                [docs[i] for i in train_indices],
                base_path / 'train/lemma_train',
                f'train{form_suffix}',
                form
            )
            DocumentProcessor.save_docs(
                [docs[i] for i in dev_indices],
                base_path / 'dev/lemma_dev',
                f'dev{form_suffix}',
                form
            )
            DocumentProcessor.save_docs(
                [docs[i] for i in test_indices],
                base_path / 'test/lemma_test',
                f'test{form_suffix}',
                form
            )

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Greek Lemmatizer Script")
    parser.add_argument("--input-dir", required=True, help="Input directory containing files to process")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--model", default="grc_proiel_trf", help="SpaCy model to use")
    parser.add_argument("--save-corrections", help="Path to save corrections JSON")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing sentences")
    parser.add_argument("--forms", nargs="+", default=["NFC"], 
                       choices=[f.name for f in NormalizationForm], help="Normalization forms to use")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Load spaCy model
        nlp = spacy.load(args.model)
        nlp.disable_pipes(["morphologizer", "tagger", "parser", "attribute_ruler"])
        
        # Load and process dictionaries
        logger.info("Loading dictionaries...")
        coda_df = pd.read_csv("../assets/NER_assets/Ancient_Words_12_5_22.csv")
        
        dictionaries = {
            'Conllu': DictionaryBuilder.build_conllu_dict("../assets/Lemmatization_training_files/Processed"),
            'Dendrosearch': DictionaryBuilder.build_dendrosearch_dict("../assets/dendrosearch_lemma_dict.txt"),
            'Coda': DictionaryBuilder.build_coda_dict(coda_df)
        }
        
        # Process INCEpTION files
        logger.info("Processing INCEpTION files...")
        inception_dict, inception_sentences = DictionaryBuilder.process_inception_files(
            Path("../assets/NER_assets/INCEpTION_files/")
        )
        dictionaries['Inception'] = inception_dict
        
        # Combine dictionaries
        logger.info("Combining dictionaries...")
        processed_pairs = DictionaryBuilder.combine_dictionaries(
            dictionaries,
            normalization_forms=args.forms,
            debug=args.debug
        )

        # Process pairs and choose most common lemma for each word
        logger.info("Processing pairs and choosing most common lemmas...")
        processed_pairs = DictionaryBuilder.process_pairs(
            processed_pairs, debug=args.debug)
        
        # Process sentences
        logger.info(f"Processing sentences with normalization forms: {', '.join(args.forms)}")
        all_corrections = {}
        all_processed_docs = {}
        total_corrections = 0
        
        # Sort forms to ensure consistent processing order
        for form in sorted(args.forms):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing sentences for {form} normalization...")
            corrections, processed_docs = SentenceProcessor.process_sentences(
                inception_sentences,
                nlp,
                processed_pairs,
                debug=args.debug,
                batch_size=args.batch_size
            )
            all_corrections[form] = corrections
            all_processed_docs[form] = processed_docs
            total_corrections += len(corrections)
            logger.info(f"Found {len(corrections)} corrections for {form}")
            
            # Save corrections if requested
            if args.save_corrections:
                corrections_file = Path(args.save_corrections)
                form_corrections_file = corrections_file.with_stem(f"{corrections_file.stem}_{form}")
                with open(form_corrections_file, 'w', encoding='utf-8') as f:
                    json.dump(corrections, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {form} corrections to {form_corrections_file}")
        
        # Split and save documents
        logger.info("Splitting and saving documents...")
        DocumentProcessor.split_and_save_docs(
            all_processed_docs[sorted(args.forms)[0]],  # Use first form as base
            Path(args.output_dir),
            forms=args.forms
        )
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info("Processing Summary:")
        logger.info(f"- Processed forms: {', '.join(args.forms)}")
        logger.info(f"- Total corrections: {total_corrections}")
        logger.info(f"- Documents saved to: {args.output_dir}")
        logger.info(f"- Model used: {args.model}")
        logger.info("Processing complete!\n")
        
    except Exception as e:
        if isinstance(e, ValueError) and "normalization form" in str(e):
            logger.error(str(e))
            sys.exit(1)
        else:
            logger.error(f"Error during execution: {e}")
            logger.debug("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
