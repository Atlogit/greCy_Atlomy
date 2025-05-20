import os
import sys
import re
import conllu
import unicodedata
import logging
from pathlib import Path

def preprocess_conllu_file(file_path):
    """
    Preprocess CoNLL-U file to ensure proper formatting
    
    Args:
        file_path (str or Path): Path to the input CoNLL-U file
    
    Returns:
        str: Preprocessed file contents
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove lines that don't match CoNLL-U token format
    processed_lines = []
    for line in lines:
        # Skip comments and empty lines
        if line.startswith('#') or line.strip() == '':
            processed_lines.append(line)
            continue
        
        # Ensure line has correct number of tab-separated fields
        fields = line.split('\t')
        if len(fields) >= 10:  # Standard CoNLL-U has 10 fields
            processed_lines.append(line)
    
    return ''.join(processed_lines)

def normalize_text(text, normalization_form='NFKD', remove_accents=False, lowercase=False, 
                   standardize_apostrophe=True, remove_extra_spaces=True, debug=False):
    """
    Normalize text with various optional transformations.
    
    Args:
        text (str): Input text to normalize
        normalization_form (str): Unicode normalization form (e.g., 'NFKD', 'NFC')
        remove_accents (bool): Remove accent marks
        lowercase (bool): Convert to lowercase
        standardize_apostrophe (bool): Replace various apostrophe-like characters
        remove_extra_spaces (bool): Remove extra whitespace
        debug (bool): Print debug information
    
    Returns:
        str: Normalized text
    """
    # Handle None or empty input
    if not text:
        return text
    
    # Normalize Unicode form first
    if normalization_form:
        text = unicodedata.normalize(normalization_form, text)
    
    # Remove accents if specified
    if remove_accents:
        text = ''.join(char for char in unicodedata.normalize('NFKD', text)
                       if unicodedata.category(char) != 'Mn')
    
    # Standardize apostrophes
    if standardize_apostrophe:
        apostrophe_map = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201B': "'",  # Single high-reversed-9 quotation mark
            '\u2032': "'",  # Prime symbol
        }
        for old, new in apostrophe_map.items():
            text = text.replace(old, new)
    
    # Remove extra spaces
    if remove_extra_spaces:
        text = ' '.join(text.split())
    
    # Lowercase if specified
    if lowercase:
        text = text.lower()
    
    # Optional debug output
    if debug:
        print(f"Original: {text}")
        print(f"Normalized: {text}")
    
    return text

def process_corpus_conllu_files(input_directory, output_directory, normalization_form='NFKD'):
    """
    Processes .conllu files in the given directory, normalizes token forms and lemmas 
    using the specified normalization form, and writes them to a new directory.
    
    Args:
        input_directory (str): Path to input directory with .conllu files
        output_directory (str): Path to output directory for processed files
        normalization_form (str): Unicode normalization form to apply
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Convert to Path objects for robust path handling
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    
    # Log input and output paths for debugging
    logger.info(f"Input Directory: {input_path.absolute()}")
    logger.info(f"Output Directory: {output_path.absolute()}")
    
    # Validate input directory
    if not input_path.is_dir():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track processing statistics
    total_files = 0
    processed_files = 0
    skipped_files = 0
    
    # Find all .conllu files, including in subdirectories
    conllu_files = list(input_path.rglob('*.conllu'))
    
    logger.info(f"Found {len(conllu_files)} .conllu files")
    
    # Process each .conllu file
    for input_file_path in conllu_files:
        total_files += 1
        
        try:
            # Preprocess the file to ensure proper formatting
            #preprocessed_content = preprocess_conllu_file(input_file_path)
            with open(input_file_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Attempt to parse the file
            try:
                sentences = conllu.parse(text)
            except Exception as parse_error:
                logger.warning(f"Parsing error in {input_file_path.name}: {parse_error}")
                skipped_files += 1
                continue
            
            # Prepare output file path
            output_file_path = output_path / f"{input_file_path.stem}_{normalization_form}.conllu"
            
            # Process and write normalized sentences
            with output_file_path.open("w", encoding="utf-8") as output_file:
                for sentence in sentences:
                    # Normalize each token's form and lemma
                    for token in sentence:
                        token["form"] = normalize_text(
                            token.get("form", ""), 
                            normalization_form, 
                            remove_accents=False, 
                            lowercase=False, 
                            standardize_apostrophe=True, 
                            remove_extra_spaces=True, 
                            debug=False
                        )
                        token["lemma"] = normalize_text(
                            token.get("lemma", ""), 
                            normalization_form, 
                            remove_accents=False, 
                            lowercase=False, 
                            standardize_apostrophe=True, 
                            remove_extra_spaces=True, 
                            debug=False
                        )
                    
                    # Write the normalized sentence
                    output_file.write(sentence.serialize())
            
            processed_files += 1
            logger.info(f"Processed file: {input_file_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing {input_file_path.name}: {e}")
            skipped_files += 1
    
    # Log processing summary
    logger.info(f"Processing complete. Total files: {total_files}, "
                f"Processed: {processed_files}, Skipped: {skipped_files}")

# Example usage with flexible path handling
if __name__ == "__main__":
    # Try multiple potential paths
    potential_paths = [
        "./assets/UD_Ancient_Greek-Perseus",
        "assets/UD_Ancient_Greek-Perseus",
        "/path/to/your/conllu/files"
    ]
    
    output_base = "../assets/UD_Ancient_Greek-Perseus_Normalized"
    
    for input_dir in potential_paths:
        if os.path.isdir(input_dir):
            print(f"Processing files from: {input_dir}")
            process_corpus_conllu_files(input_dir, output_base, normalization_form='NFKD')
            break
    else:
        print("No valid input directory found. Please specify the correct path.")
        sys.exit(1)