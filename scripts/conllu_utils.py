#!/usr/bin/env python3
"""
Utilities for processing CONLLU format files.
"""

import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import subprocess

import conllu
from tqdm import tqdm

from .utils import TextNormalizer, NormalizationForm

logger = logging.getLogger(__name__)

class ConlluProcessor:
    """Handles CONLLU file processing and conversion."""
    
    def __init__(self):
        """Initialize ConlluProcessor."""
        self.normalizer = TextNormalizer()

    def process_sentences(self, 
                        input_files: List[str],
                        output_file: str,
                        combine: bool = False,
                        show_apostrophe_changes: bool = False) -> None:
        """
        Process CONLLU files: clean text, separate sentences based on punctuation.
        
        Args:
            input_files (List[str]): List of paths to input CONLLU files
            output_file (str): Path to the output CONLLU file
            combine (bool): Whether to combine input files
            show_apostrophe_changes (bool): Whether to show apostrophe changes
        """
        all_sentences = []
        total_apostrophe_changes = {}

        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()
                
            sentences = conllu.parse(text)
            if combine:
                all_sentences.extend(sentences)
            else:
                all_sentences = sentences
                break

        rebuilt_sentences = []
        sent_id = 1
        current_sentence_tokens = []
        token_id = 1

        for sentence in all_sentences:
            for token in sentence:
                # Clean text
                before_form = token['form']
                before_lemma = token['lemma']

                token['form'] = self.normalizer.normalize_text(
                    token['form'],
                    standardize_apostrophe=True,
                    remove_extra_spaces=True
                )
                token['lemma'] = self.normalizer.normalize_text(
                    token['lemma'],
                    standardize_apostrophe=True,
                    remove_extra_spaces=True
                )

                # Track apostrophe changes
                if show_apostrophe_changes:
                    for apos in self.normalizer.APOSTROPHES:
                        if apos != self.normalizer.CORRECT_APOSTROPHE:
                            form_count = before_form.count(apos)
                            if form_count > 0:
                                total_apostrophe_changes[apos] = total_apostrophe_changes.get(apos, 0) + form_count
                            lemma_count = before_lemma.count(apos)
                            if lemma_count > 0:
                                total_apostrophe_changes[apos] = total_apostrophe_changes.get(apos, 0) + lemma_count

                token['id'] = token_id
                current_sentence_tokens.append(token)
                token_id += 1

                # Check for sentence end
                if token["form"] in [".", "·"]:
                    metadata = {"sent_id": str(sent_id), "text": "NA"}
                    rebuilt_sentences.append(conllu.TokenList(tokens=current_sentence_tokens, metadata=metadata))
                    current_sentence_tokens = []
                    sent_id += 1
                    token_id = 1

        # Handle last sentence
        if current_sentence_tokens:
            metadata = {"sent_id": str(sent_id), "text": "NA"}
            rebuilt_sentences.append(conllu.TokenList(tokens=current_sentence_tokens, metadata=metadata))

        # Write output
        with open(output_file, "w", encoding="utf-8") as out_f:
            for sentence in rebuilt_sentences:
                out_f.write(sentence.serialize())
                out_f.write("\n\n")

        # Print summary
        summary = "Combined" if combine else "Original"
        logger.info(f"{summary} number of sentences from input files: {len(all_sentences)}")
        logger.info(f"Rebuilt and separated {len(rebuilt_sentences)} sentences.")

        if show_apostrophe_changes and total_apostrophe_changes:
            total_replaced = sum(total_apostrophe_changes.values())
            logger.info(f"\nApostrophe changes for {input_files[0]}:")
            for apos, count in total_apostrophe_changes.items():
                logger.info(f"  {apos} -> {self.normalizer.CORRECT_APOSTROPHE}: {count}")
            logger.info(f"  Total replaced: {total_replaced}")

    def adjust_tokens_for_spacy(self, sentences: List[Any], debug: bool = False) -> List[Any]:
        """
        Adjust tokens for spaCy's requirements.
        
        Args:
            sentences: List of CONLLU sentences
            debug (bool): Enable debug logging
            
        Returns:
            List[Any]: Adjusted sentences
        """
        for sentence in sentences:
            for token in sentence:
                # Adjust forms and lemmas
                if token["form"] in ['', "_", '—', '-']:
                    token["form"] = token["lemma"] if token["lemma"] not in ['', "_", '—', '-'] else "_"
                    if debug:
                        logger.debug(f"Adjusted form for token {token['id']}, form: {token['form']}")

                if token["lemma"] in ['', "_", '—', '-']:
                    token["lemma"] = "_"
                    if debug:
                        logger.debug(f"Adjusted lemma for token {token['id']}, lemma: {token['lemma']}")

                # ID and UPOS adjustments
                if token["id"] == '':
                    token["id"] = "UNK"
                    if debug:
                        logger.debug(f"Adjusted ID for token {token['form']}, ID: {token['id']}")

                if token["upos"] in ['', "_", '—', '-']:
                    token["upos"] = "_"
                    if debug:
                        logger.debug(f"Adjusted UPOS for token {token['form']}, UPOS: {token['upos']}")

                if token["upos"] in ['END', 'MID']:
                    token["upos"] = "NOUN"
                    if debug:
                        logger.debug(f"Adjusted UPOS for token {token['form']}, UPOS: {token['upos']}")

        return sentences

    def validate_head_indices(self, sentences: List[Any], debug: bool = False) -> bool:
        """
        Validate head indices in sentences.
        
        Args:
            sentences: List of CONLLU sentences
            debug (bool): Enable debug logging
            
        Returns:
            bool: Whether all head indices are valid
        """
        for sentence in sentences:
            token_ids = {token["id"] for token in sentence}
            for token in sentence:
                head = token.get("head")
                
                if head is None:
                    if debug:
                        logger.debug(f"Missing head for token '{token['form']}' in sentence: {sentence.metadata.get('text', 'NA')}")
                    return False

                if head not in token_ids and head != 0:
                    if debug:
                        logger.debug(f"Invalid head index {head} for token '{token['form']}' in sentence: {sentence.metadata.get('text', 'NA')}")
                    return False
        
        return True

    def convert_to_spacy(self, file_path: str, output_directory: str, sentences: List[Any]) -> None:
        """
        Convert CONLLU file to spaCy format.
        
        Args:
            file_path (str): Path to CONLLU file
            output_directory (str): Output directory
            sentences: List of CONLLU sentences
        """
        extra_args = "--n-sents 10" if len(sentences) >= 10 else ""
        convert_command = f"python -m spacy convert {file_path} {output_directory} -c conllu -m --merge-subtokens {extra_args}"
        
        try:
            subprocess.run(convert_command.split(), check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting file '{Path(file_path).name}': {e}")

    def process_and_normalize_files(self,
                                 input_directory: str,
                                 output_directory: str,
                                 normalization_form: str = 'NFC',
                                 debug: bool = False) -> None:
        """
        Process and normalize CONLLU files.
        
        Args:
            input_directory (str): Input directory path
            output_directory (str): Output directory path
            normalization_form (str): Unicode normalization form
            debug (bool): Enable debug logging
        """
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        for file_path in Path(input_directory).glob("*.conllu"):
            if debug:
                logger.debug(f"Processing {file_path.name}")

            # Read and parse file
            sentences = conllu.parse(file_path.read_text(encoding="utf-8"))
            
            # Adjust tokens
            sentences = self.adjust_tokens_for_spacy(sentences)
            
            # Process sentences
            for sentence in sentences:
                for token in sentence:
                    token["form"] = self.normalizer.normalize_text(
                        token["form"],
                        form=normalization_form
                    )
                    token["lemma"] = self.normalizer.normalize_text(
                        token["lemma"],
                        form=normalization_form
                    )

            # Split into train/dev
            random.seed(42)
            random.shuffle(sentences)
            split_index = int(len(sentences) * 0.9)
            train_data, dev_data = sentences[:split_index], sentences[split_index:]

            # Save splits
            base_name = file_path.stem
            train_path = Path(output_directory) / f"{base_name}_{normalization_form}_train.conllu"
            dev_path = Path(output_directory) / f"{base_name}_{normalization_form}_dev.conllu"

            with open(train_path, "w", encoding="utf-8") as f:
                for sentence in train_data:
                    f.write(sentence.serialize())

            with open(dev_path, "w", encoding="utf-8") as f:
                for sentence in dev_data:
                    f.write(sentence.serialize())

            if debug:
                logger.debug(f"Processed and normalized {file_path.name}. Train and dev data saved.")