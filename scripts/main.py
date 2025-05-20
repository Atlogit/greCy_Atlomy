#!/usr/bin/env python3
"""
Main entry point for Greek text preprocessing pipeline.
"""

import logging
import argparse
from pathlib import Path
from typing import List, Optional

import spacy
from spacy.tokens import DocBin

from .utils import TextNormalizer, NormalizationForm
from .preprocess import Preprocessor
from .lemma_preprocess import SentenceProcessor
from .morphology_preprocess import MorphologyProcessor
from .spancat_preprocess import SpanProcessor
from .analysis_utils import LemmaEvaluator, NEREvaluator, SpanCatEvaluator

logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

def load_model(model_path: str, components: Optional[List[str]] = None) -> spacy.language.Language:
    """
    Load spaCy model and configure components.
    
    Args:
        model_path: Path to spaCy model
        components: Optional list of components to disable
        
    Returns:
        Loaded spaCy model
    """
    try:
        nlp = spacy.load(model_path)
        if components:
            for component in components:
                if component in nlp.pipe_names:
                    nlp.disable_pipe(component)
        return nlp
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def process_pipeline(args: argparse.Namespace) -> None:
    """
    Run the preprocessing pipeline.
    
    Args:
        args: Command line arguments
    """
    try:
        # Load model
        nlp = load_model(args.model)
        
        # Initialize processors
        normalizer = TextNormalizer()
        preprocessor = Preprocessor(nlp)
        sentence_processor = SentenceProcessor(nlp)
        morphology_processor = MorphologyProcessor(nlp)
        span_processor = SpanProcessor(nlp)
        
        # Process each requested step
        for step in args.steps:
            logger.info(f"Processing step: {step}")
            
            if step == "lemma":
                sentence_processor.process_directory(
                    Path(args.input_dir) / "lemma",
                    Path(args.output_dir) / "lemma",
                    form=NormalizationForm.from_string(args.form),
                    debug=args.debug
                )
            
            elif step == "morphology":
                morphology_processor.process_directory(
                    Path(args.input_dir) / "morphology",
                    Path(args.output_dir) / "morphology",
                    form=NormalizationForm.from_string(args.form),
                    debug=args.debug
                )
            
            elif step == "spancat":
                span_processor.process_directory(
                    Path(args.input_dir) / "spancat",
                    Path(args.output_dir) / "spancat",
                    form=NormalizationForm.from_string(args.form),
                    debug=args.debug
                )
            
            else:
                logger.warning(f"Unknown step: {step}")
                
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        if args.debug:
            logger.debug("Stack trace:", exc_info=True)
        raise

def evaluate_models(args: argparse.Namespace) -> None:
    """
    Evaluate model performance.
    
    Args:
        args: Command line arguments
    """
    try:
        # Load models
        models = [load_model(path) for path in args.model_paths]
        
        # Load test data
        test_docs = DocBin().from_disk(args.test_data)
        docs = list(test_docs.get_docs(models[0].vocab))
        
        # Initialize evaluators
        if args.eval_type == "lemma":
            evaluator = LemmaEvaluator(
                models,
                model_names=args.model_names,
                norm_method=args.form
            )
            results = evaluator.evaluate_lemmas(
                docs,
                evaluate_lemma=True,
                evaluate_pos=args.eval_pos,
                evaluate_morph=args.eval_morph
            )
            
        elif args.eval_type == "ner":
            evaluator = NEREvaluator(
                models,
                model_names=args.model_names,
                norm_method=args.form
            )
            results = evaluator.evaluate_ner(docs)
            
        elif args.eval_type == "spancat":
            evaluator = SpanCatEvaluator(
                models,
                model_names=args.model_names,
                norm_method=args.form
            )
            results = evaluator.evaluate_spans(docs)
            
        else:
            logger.error(f"Unknown evaluation type: {args.eval_type}")
            return
        
        # Save results if requested
        if args.output:
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.debug:
            logger.debug("Stack trace:", exc_info=True)
        raise

def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Greek Text Preprocessing Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process pipeline command
    process_parser = subparsers.add_parser("process", help="Run preprocessing pipeline")
    process_parser.add_argument("--input-dir", required=True, help="Input directory")
    process_parser.add_argument("--output-dir", required=True, help="Output directory")
    process_parser.add_argument("--steps", nargs="+", default=["lemma"],
                              choices=["lemma", "morphology", "spancat"],
                              help="Preprocessing steps to run")
    process_parser.add_argument("--model", default="grc_proiel_trf",
                              help="SpaCy model to use")
    process_parser.add_argument("--form", default="NFC",
                              choices=[f.name for f in NormalizationForm],
                              help="Normalization form to use")
    process_parser.add_argument("--debug", action="store_true",
                              help="Enable debug logging")
    
    # Evaluate models command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model-paths", nargs="+", required=True,
                           help="Paths to models to evaluate")
    eval_parser.add_argument("--model-names", nargs="+",
                           help="Names for models in output")
    eval_parser.add_argument("--test-data", required=True,
                           help="Path to test data")
    eval_parser.add_argument("--eval-type", required=True,
                           choices=["lemma", "ner", "spancat"],
                           help="Type of evaluation to perform")
    eval_parser.add_argument("--form", default="NFC",
                           choices=[f.name for f in NormalizationForm],
                           help="Normalization form to use")
    eval_parser.add_argument("--eval-pos", action="store_true",
                           help="Evaluate POS tagging (lemma only)")
    eval_parser.add_argument("--eval-morph", action="store_true",
                           help="Evaluate morphological analysis (lemma only)")
    eval_parser.add_argument("--output",
                           help="Path to save evaluation results")
    eval_parser.add_argument("--debug", action="store_true",
                           help="Enable debug logging")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    if args.command == "process":
        process_pipeline(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()