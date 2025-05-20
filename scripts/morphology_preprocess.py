#!/usr/bin/env python3
"""
Morphology and POS-specific preprocessing module.
"""

import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

import spacy
from spacy.tokens import Doc, MorphAnalysis
from tqdm import tqdm

from .utils import TextNormalizer, NormalizationForm
from .file_io import FileIO
from .preprocess import Preprocessor
from .conllu_utils import ConlluProcessor

logger = logging.getLogger(__name__)

class MorphologyProcessor:
    """Handles morphology and POS processing."""
    
    def __init__(self, nlp: spacy.language.Language):
        """
        Initialize morphology processor.
        
        Args:
            nlp: Loaded spaCy model
        """
        self.nlp = nlp
        self.normalizer = TextNormalizer()
        self.conllu_processor = ConlluProcessor()

    def set_token_attributes(self, doc: Doc, sentence_data: List[Dict[str, Any]], debug: bool = False) -> Doc:
        """
        Set token attributes including POS tags, lemmas, morphological features, and dependencies.
        
        Args:
            doc (Doc): spaCy Doc to modify
            sentence_data (List[Dict[str, Any]]): Token data from CONLLU
            debug (bool): Enable debug logging
            
        Returns:
            Doc: Modified spaCy Doc
        """
        root_token = None

        # First pass: Set basic attributes and identify root
        for i, token in enumerate(doc):
            t = sentence_data[i]
            
            # Handle punctuation
            if t['form'] in self.normalizer.PUNCTUATION:
                doc[i].pos_ = 'PUNCT'
            else:
                # Set POS tags
                if t['upos'] in ['', '_', '—', '-', 'X', 'END', 'MID']:
                    doc[i].pos_ = ''
                else:
                    doc[i].pos_ = t['upos']

                # Set fine-grained POS tags (xpos)
                if t['xpos'] in ['', '_', '—', '-', 'X', 'END', 'MID']:
                    doc[i].tag_ = ''
                else:
                    doc[i].tag_ = t['xpos']

            # Set lemmas
            doc[i].lemma_ = '' if t['lemma'] in ['', '_', '—', '-'] else t['lemma']

            # Set morphological features
            if t['feats']:
                morph_analysis_hash = self.nlp.vocab.morphology.add(t['feats'])
                doc[i].morph = MorphAnalysis.from_id(self.nlp.vocab, morph_analysis_hash)

            # Set dependency labels
            if t['deprel'] in ['', '_', '—', '-']:
                doc[i].dep_ = 'None'
            else:
                doc[i].dep_ = t['deprel']
                if t['deprel'] == 'root':
                    root_token = doc[i]
                    if debug:
                        logger.debug(f"Found root token: {root_token.text}")

        # Second pass: Set head relationships
        for i, token in enumerate(doc):
            t = sentence_data[i]
            
            # Handle head relationships
            if t['head'] not in [None, '', '_', '—', '-']:
                head_idx = int(t['head']) - 1
                if 0 <= head_idx < len(doc):
                    doc[i].head = doc[head_idx]
                    # Ensure head token has POS tag
                    if not doc[i].head.pos_:
                        doc[i].head.pos_ = doc[head_idx].pos_ if doc[head_idx].pos_ else ''
                    if debug:
                        logger.debug(f"Set head for token '{token.text}' to '{doc[i].head.text}'")
            else:
                # If no head is specified and this isn't the root, set root as head
                if root_token and token != root_token:
                    doc[i].head = root_token
                    if debug:
                        logger.debug(f"Set default head (root) for token '{token.text}'")

        return doc

    def process_sentences(self, 
                        sentences_data: List[Dict[str, Any]],
                        debug: bool = False) -> List[Doc]:
        """
        Process sentences into spaCy Docs with morphological annotations.
        
        Args:
            sentences_data: List of sentence data from CONLLU
            debug (bool): Enable debug logging
            
        Returns:
            List[Doc]: List of processed spaCy Docs
        """
        docs = []
        for sentence in sentences_data:
            # Filter out empty tokens
            tokens = [t for t in sentence if t['form']]
            
            # Create words and spaces lists
            words = [t['form'] for t in tokens]
            spaces = [not (t['misc'] and t['misc'].get('SpaceAfter') == 'No') 
                     for t in tokens]
            
            # Create base doc
            doc = Doc(self.nlp.vocab, words=words, spaces=spaces)
            
            # Set token attributes
            doc = self.set_token_attributes(doc, tokens, debug=debug)
            
            docs.append(doc)
            
            if debug:
                self._validate_doc(doc)
        
        return docs

    def _validate_doc(self, doc: Doc) -> None:
        """
        Validate document annotations and log any issues.
        
        Args:
            doc (Doc): spaCy Doc to validate
        """
        for token in doc:
            if not token.pos_:
                logger.warning(f"Empty POS tag for token '{token.text}' in '{doc.text}'")
            if not token.dep_:
                logger.warning(f"Empty dependency label for token '{token.text}' in '{doc.text}'")
            if token.head == token and token.dep_ != "ROOT":
                logger.warning(f"Self-referential head for non-root token '{token.text}' in '{doc.text}'")

    def process_file(self,
                    file_path: Path,
                    form: NormalizationForm = NormalizationForm.NFC,
                    debug: bool = False) -> List[Doc]:
        """
        Process a CONLLU file into spaCy Docs.
        
        Args:
            file_path (Path): Path to CONLLU file
            form (NormalizationForm): Unicode normalization form
            debug (bool): Enable debug logging
            
        Returns:
            List[Doc]: List of processed spaCy Docs
        """
        # Parse CONLLU file
        sentences = self.conllu_processor.read_and_parse_conllu(file_path, debug=debug)
        
        # Normalize text
        for sentence in sentences:
            for token in sentence:
                token['form'] = self.normalizer.normalize_text(
                    token['form'],
                    form=form,
                    standardize_apostrophe=True
                )
                if token['lemma'] not in ['', '_', '—', '-']:
                    token['lemma'] = self.normalizer.normalize_text(
                        token['lemma'],
                        form=form,
                        standardize_apostrophe=True,
                        lowercase=True
                    )
        
        # Process sentences
        return self.process_sentences(sentences, debug=debug)

    def process_directory(self,
                        input_dir: Path,
                        output_dir: Path,
                        form: NormalizationForm = NormalizationForm.NFC,
                        debug: bool = False) -> None:
        """
        Process all CONLLU files in a directory.
        
        Args:
            input_dir (Path): Input directory containing CONLLU files
            output_dir (Path): Output directory for processed files
            form (NormalizationForm): Unicode normalization form
            debug (bool): Enable debug logging
        """
        docs = []
        
        # Process each CONLLU file
        for file_path in tqdm(list(input_dir.glob("*.conllu")), desc="Processing CONLLU files"):
            if debug:
                logger.debug(f"Processing {file_path.name}")
            
            try:
                file_docs = self.process_file(file_path, form=form, debug=debug)
                docs.extend(file_docs)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                if debug:
                    logger.debug("Stack trace:", exc_info=True)
                continue
        
        if not docs:
            logger.warning(f"No documents processed from {input_dir}")
            return
        
        # Save processed docs
        output_dir.mkdir(parents=True, exist_ok=True)
        FileIO.save_docs(docs, output_dir / f"morphology_{form.name.lower()}.spacy")
        logger.info(f"Saved {len(docs)} documents to {output_dir}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Morphology and POS Preprocessing")
    parser.add_argument("--input-dir", required=True, help="Input directory containing CONLLU files")
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
        nlp.disable_pipes(["lemmatizer", "ner"])  # Keep morphologizer and parser
        
        # Initialize processor
        processor = MorphologyProcessor(nlp)
        
        # Process directory
        processor.process_directory(
            Path(args.input_dir),
            Path(args.output_dir),
            form=NormalizationForm.from_string(args.form),
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()