#!/usr/bin/env python3
"""
Lemmatization-specific preprocessing module.
"""

import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

import spacy
import pandas as pd
from tqdm import tqdm
from spacy.tokens import Doc

from .utils import TextNormalizer, NormalizationForm
from .dictionary_utils import DictionaryBuilder
from .file_io import FileIO
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)

class SentenceProcessor:
    """Handles sentence processing and lemmatization."""
    
    def __init__(self, nlp: spacy.language.Language):
        """
        Initialize sentence processor.
        
        Args:
            nlp: Loaded spaCy model
        """
        self.nlp = nlp
        self.normalizer = TextNormalizer()

    def process_sentences(self,
                        sentences_list: List[Tuple[str, str]], 
                        processed_pairs: Dict[NormalizationForm, Dict[str, Dict[str, Set[str]]]],
                        debug: bool = False,
                        batch_size: int = 1000) -> Tuple[List[Dict], List[Doc]]:
        """
        Process sentences using spaCy and apply lemmatization corrections.
        
        Args:
            sentences_list: List of (sentence, source) tuples
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
                self.normalizer.normalize_text(
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
        for doc, meta in tqdm(zip(self.nlp.pipe(sentences_for_processing, batch_size=batch_size), metadata),
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

def modify_token_attributes(doc: Doc, debug: bool = False) -> Doc:
    """
    Modify token attributes according to specific rules.
    
    Args:
        doc (Doc): spaCy Doc to modify
        debug (bool): Enable debug logging
        
    Returns:
        Doc: Modified spaCy Doc
    """
    for token in doc:
        # Set empty lemma for trainable lemmatizer
        if token.lemma_ in ['', "_", '—', '-']:
            token.lemma_ = ''
            
        # Set empty POS tags
        if token.pos_ in ['', "_", '—', '-', 'X', 'END', 'MID']:
            token.pos_ = ""
            
        # Set dependency labels
        if token.dep_ in ['', "_", '—', '-']:
            token.dep_ = "None"
            
        if token.head.dep_ in ['', "_", '—', '-']:
            token.head.dep_ = "None"
            
    return doc

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Greek Lemmatizer Preprocessing")
    parser.add_argument("--input-dir", required=True, help="Input directory containing files to process")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--model", default="grc_proiel_trf", help="SpaCy model to use")
    parser.add_argument("--save-corrections", help="Path to save corrections JSON")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--forms", nargs="+", default=["NFC"], 
                       choices=[f.name for f in NormalizationForm], help="Normalization forms to use")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Load spaCy model
        nlp = spacy.load(args.model)
        nlp.disable_pipes(["morphologizer", "tagger", "parser", "attribute_ruler"])
        
        # Initialize processors
        sentence_processor = SentenceProcessor(nlp)
        
        # Load and process dictionaries
        logger.info("Loading dictionaries...")
        coda_df = pd.read_csv(Path(args.input_dir) / "Ancient_Words.csv")
        
        dictionaries = {
            'Conllu': DictionaryBuilder.build_conllu_dict(str(Path(args.input_dir) / "conllu")),
            'Dendrosearch': DictionaryBuilder.build_dendrosearch_dict(str(Path(args.input_dir) / "dendrosearch_lemma_dict.txt")),
            'Coda': DictionaryBuilder.build_coda_dict(coda_df)
        }
        
        # Process INCEpTION files
        logger.info("Processing INCEpTION files...")
        inception_dict, inception_sentences = DictionaryBuilder.process_inception_files(
            Path(args.input_dir) / "inception"
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
        processed_pairs = DictionaryBuilder.process_pairs(processed_pairs, debug=args.debug)
        
        # Process sentences
        logger.info(f"Processing sentences with normalization forms: {', '.join(args.forms)}")
        all_corrections = {}
        all_processed_docs = {}
        total_corrections = 0
        
        # Sort forms to ensure consistent processing order
        for form in sorted(args.forms):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing sentences for {form} normalization...")
            corrections, processed_docs = sentence_processor.process_sentences(
                inception_sentences,
                processed_pairs,
                debug=args.debug,
                batch_size=args.batch_size
            )
            
            # Modify token attributes
            processed_docs = [modify_token_attributes(doc, debug=args.debug) for doc in processed_docs]
            
            all_corrections[form] = corrections
            all_processed_docs[form] = processed_docs
            total_corrections += len(corrections)
            logger.info(f"Found {len(corrections)} corrections for {form}")
            
            # Save corrections if requested
            if args.save_corrections:
                corrections_file = Path(args.save_corrections)
                form_corrections_file = corrections_file.with_stem(f"{corrections_file.stem}_{form}")
                FileIO.write_json(corrections, form_corrections_file)
                logger.info(f"Saved {form} corrections to {form_corrections_file}")
        
        # Save processed docs
        output_dir = Path(args.output_dir)
        for form in args.forms:
            form_dir = output_dir / form.lower()
            FileIO.ensure_dir(form_dir)
            FileIO.save_docs(all_processed_docs[form], form_dir / "processed.spacy")
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info("Processing Summary:")
        logger.info(f"- Processed forms: {', '.join(args.forms)}")
        logger.info(f"- Total corrections: {total_corrections}")
        logger.info(f"- Documents saved to: {args.output_dir}")
        logger.info(f"- Model used: {args.model}")
        logger.info("Processing complete!\n")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()