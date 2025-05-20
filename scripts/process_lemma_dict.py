import json
import unicodedata
import re
from pathlib import Path

def clean_and_remove_accents(text: str) -> str:
    """
    Cleans the given text by removing diacritics (accents), except for specific characters.
    """
    allowed_characters = [' ̓', "᾿", "᾽", "'", "'", "'", 'ʼ', '̓']  # Including Greek apostrophe
    
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
        
    try:
        non_accent_chars = [c for c in unicodedata.normalize('NFKD', text) 
                          if unicodedata.category(c) != 'Mn' or c in allowed_characters]
        return ''.join(non_accent_chars)
    except Exception as e:
        print(f"An error occurred: {e}")
        return text

def normalize_text(text: str, form: str = 'NFKD',
                  remove_accents: bool = False,
                  lowercase: bool = False,
                  standardize_apostrophe: bool = True,
                  remove_brackets: bool = False,
                  remove_trailing_numbers: bool = False) -> str:
    """
    Applies multiple text normalization and cleaning steps on the input text.
    """
    # Define standard apostrophe characters
    apostrophes = ["᾽", "᾿", "'", "’", "‘"]
    correct_apostrophe = "ʼ"
    
    normalized_text = text

    # Standardize apostrophes if required
    if standardize_apostrophe:
        for apos in apostrophes:
            normalized_text = normalized_text.replace(apos, correct_apostrophe)
            
    # Remove accents if required
    if remove_accents:
        try:
            normalized_text = clean_and_remove_accents(normalized_text)
        except Exception as e:
            print(f"An error occurred while removing accents: {e}")
            return text
            
    # Convert to lowercase if required
    if lowercase:
        normalized_text = normalized_text.lower()
        
    # Unicode normalization
    if form in ('NFC', 'NFD', 'NFKC', 'NFKD'):
        normalized_text = unicodedata.normalize(form, normalized_text)
            
    # Remove brackets if required
    if remove_brackets:
        normalized_text = re.sub(r'[\(\)\[\]]', '', normalized_text)
        
    # Remove trailing numbers if required
    if remove_trailing_numbers:
        normalized_text = re.sub(r'^\d+|\d+$', '', normalized_text)
            
    return normalized_text

def load_lemma_dict(filename: str) -> dict:
    """
    Load the preprocessed lemma dictionary from JSON
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_word(word: str, form: str = 'NFKD') -> str:
    """
    Normalize a word according to the standard rules for word forms
    """
    return normalize_text(
        text=word,
        form=form,
        remove_accents=False,
        lowercase=False, 
        standardize_apostrophe=True,
        remove_brackets=True,
        remove_trailing_numbers=True
    )

def normalize_lemma(word: str, form: str = 'NFKD') -> str:
    """
    Normalize a word according to the standard rules for lemma forms
    """
    return normalize_text(
        text=word,
        form=form,
        remove_accents=False,
        lowercase=True,
        standardize_apostrophe=True, 
        remove_brackets=True,
        remove_trailing_numbers=True
    )

if __name__ == "__main__":
    # Example usage
    lemma_dict = load_lemma_dict("../assets/processed_coda_lemmas.json")
    
    # Example word normalization
    word = "ἄνθρωπος)"
    normalized_word = normalize_word(word)
    normalized_lemma = normalize_lemma(word)
    print(f"Original: {word}")
    print(f"Normalized word: {normalized_word}")
    print(f"Normalized lemma: {normalized_lemma}")