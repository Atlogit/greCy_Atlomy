import os
import zipfile
import tempfile
from pathlib import Path
from cassis import *
import logging

# Set up logging
logger = logging.getLogger(__name__)

TYPESYSTEM_FILENAME = "TypeSystem.xml"
INCEPTION_TRAIN_DATA = []

def extract_and_process_zip_files(input_path, form=None, debug=None, debug_inception=None):
    """
    Extract and process zip files with individual typesystem loading for each zip.
    
    Args:
        input_path (str): Path to the zip file
        form (str, optional): Form for text normalization
        debug (bool, optional): Enable debug mode
        debug_inception (bool, optional): Enable inception-specific debug mode
    """
    # Apply normalization as per configurations or provided form
    form = form if form is not None else app_config.DEFAULT_FORM
    debug = app_config.DEBUG

    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        
        # Extract the zip file
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(tempdir)
            
        # Get the basename of the .zip file for use as an identifier
        base_zip_filename = os.path.splitext(os.path.basename(input_path))[0]
        
        # Find and load the typesystem for this specific zip
        typesystem_path = tempdir_path / TYPESYSTEM_FILENAME
        if not typesystem_path.exists():
            logger.error(f"No TypeSystem.xml found in {input_path}")
            return
            
        try:
            with open(typesystem_path, 'rb') as f:
                typesystem = load_typesystem(f)
                
            # Process each XMI file with this zip's specific typesystem
            for f in os.listdir(tempdir):
                if f.endswith(".xmi"):
                    file_identifier = f"{os.path.splitext(f)[0]}"
                    process_xmi_file(
                        os.path.join(tempdir, f), 
                        typesystem, 
                        file_identifier, 
                        form, 
                        debug, 
                        debug_inception
                    )
                    
        except Exception as e:
            logger.error(f"Error processing zip file {input_path}: {e}")

def process_xmi_file(filename, typesystem, file_identifier, form=None, debug=None, debug_inception=None):
    """
    Process individual XMI file with its specific typesystem.
    
    Args:
        filename (str): Path to XMI file
        typesystem: The typesystem to use for this file
        file_identifier (str): Identifier for the file
        form (str, optional): Form for text normalization
        debug (bool, optional): Enable debug mode
        debug_inception (bool, optional): Enable inception-specific debug mode
    """
    # Apply normalization as per configurations or provided form
    form = form if form is not None else app_config.DEFAULT_FORM
    debug = app_config.DEBUG

    try:
        with open(filename, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=typesystem, lenient=True)
            logger.debug(f"Processing file: {filename} as {file_identifier}") if debug_inception else None
            process_cas(cas, file_identifier, form, debug, debug_inception)
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")

def process_cas(cas, file_identifier, form=None, debug=None, debug_inception=None):
    """
    Process CAS object and extract sentences and entities.
    
    Args:
        cas: The CAS object to process
        file_identifier (str): Identifier for the source file
        form (str, optional): Form for text normalization
        debug (bool, optional): Enable debug mode
        debug_inception (bool, optional): Enable inception-specific debug mode
    """
    # Apply normalization as per configurations or provided form
    form = form if form is not None else app_config.DEFAULT_FORM
    debug = app_config.DEBUG

    for sentence in cas.select(("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")):
        process_sentence(sentence, cas, file_identifier, form, debug, debug_inception)

def calculate_token_positions(normalized_sentence_text, adjusted_token_start, normalized_token_text, last_match_end, debug=None):
    """
    Attempts to adjust token positions from original indices considering the normalized sentence text.
    
    Args:
        normalized_sentence_text (str): The full sentence text after normalization
        adjusted_token_start (int): Original start index of the token before normalization
        normalized_token_text (str): The specific token text after normalization
        last_match_end (int): End position of the last matched token
        debug (bool, optional): Enable debug mode
    
    Returns:
        tuple: Adjusted start and end indices of the token in the normalized sentence
    """
    debug = debug if debug is not None else app_config.DEBUG

    # Validate input data
    if not normalized_sentence_text:
        logger.error("Normalized sentence text is empty.")
        return None

    # Attempt to find the closest match of the normalized token in the normalized text
    try:
        window_size = len(normalized_token_text)
        start_search = last_match_end if last_match_end is not None else 0
        new_token_begin = normalized_sentence_text.find(normalized_token_text, start_search)

        if new_token_begin == -1:
            logger.warning(f"Token '{normalized_token_text}' not found in the normalized text after position {start_search}.")
            return None

        new_token_end = new_token_begin + window_size
    
        logger.debug(f"Adjusted indices: {new_token_begin}-{new_token_end} for token '{normalized_token_text}' in normalized text.")
        return new_token_begin, new_token_end
    except Exception as e:
        logger.error(f"Unexpected error occurred while calculating token positions: {e}")
        return None

def process_sentence(sentence, cas, file_identifier, form=None, debug=None, debug_inception=None, **kwargs):
    """
    Process each sentence from the CAS file, extract entity tokens along with labels,
    calculate corrected token positions based on the cleaned sentence,
    and append to INCEPTION_TRAIN_DATA for NER training.
    
    Args:
        sentence: The sentence to process
        cas: The CAS object containing the sentence
        file_identifier (str): Identifier for the source file
        form (str, optional): Form for text normalization
        debug (bool, optional): Enable debug mode
        debug_inception (bool, optional): Enable inception-specific debug mode
        **kwargs: Additional keyword arguments for text normalization
    """
    # Apply normalization as per configurations or provided form
    debug = debug if debug is not None else app_config.DEBUG
    # Ensuring 'remove_accents' and 'lowercase' have fixed values for this function's purpose
    kwargs['remove_accents'] = False
    kwargs['lowercase'] = True  # Override/debug settings are explicitly set here
    kwargs['remove_extra_spaces'] = True
    kwargs["remove_brackets"] = False
    kwargs["form"] = form
    logger.debug("kwargs: %s", kwargs) if debug else None
    
    # Normalize inputs
    original_sentence_text = sentence.get_covered_text()
    normalized_sentence_text = normalize_text(original_sentence_text, **{**app_config.__dict__, **kwargs})
    
    sentence_start_offset = sentence.begin  # Sentence start relative to the full document

    logger.debug("Final FORM: %s", form) if debug_inception else None
    logger.debug("original sentence: %s", original_sentence_text) if debug_inception else None
    logger.debug("normalized_sentence_text: %s", normalized_sentence_text) if debug_inception else None
    
    spans = []
    last_match_end = None
    for token in cas.select_covered('webanno.custom.CategoryType', sentence):
        # Adjust token indices to be relative to the start of the sentence
        adjusted_token_start = token.begin - sentence_start_offset
        adjusted_token_end = token.end - sentence_start_offset
        logger.debug(f"Token index: {token.begin}-{token.end}, Adjusted index for sentence: {adjusted_token_start}-{adjusted_token_end}, Token text: {token.get_covered_text()}") if debug_inception else None
        
        normalized_token_text = normalize_text(token.get_covered_text(), **{**app_config.__dict__, **kwargs})
        token_positions = calculate_token_positions(normalized_sentence_text, adjusted_token_start, normalized_token_text, last_match_end, debug=debug_inception)
        if token_positions is None:
            logger.error(f"Error calculating positions for token '{token.get_covered_text()}' in sentence '{original_sentence_text}'")
            continue
        
        new_token_begin, new_token_end = token_positions
        spans.append((new_token_begin, new_token_end, token.value("Value")))
        last_match_end = new_token_end

    if spans:
        INCEPTION_TRAIN_DATA.append((normalized_sentence_text, {'entities': spans}, file_identifier))
        logger.debug("entities: %s", spans) if debug_inception else None
        logger.debug("normalized sentence: %s", normalized_sentence_text) if debug_inception else None
    return INCEPTION_TRAIN_DATA