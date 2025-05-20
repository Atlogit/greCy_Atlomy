import pandas as pd
import json
from pathlib import Path
import unicodedata
import re
from typing import Literal

def clean_and_remove_accents(text: str) -> str:
    """
    Cleans the given text by removing diacritics (accents), except for specific characters.
    """
    allowed_characters = [' ̓', "᾿", "᾽", "'", "'", "'", 'ʼ', '̓']  # Including the Greek apostrophe
    
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
                  remove_trailing_numbers: bool = False,
                  debug: bool = False) -> str:
    """
    Applies multiple text normalization and cleaning steps on the input text.
    
    Parameters:
    - text (str): The text to be normalized
    - form (str): Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
    - remove_accents (bool): If True, removes diacritical marks
    - lowercase (bool): If True, converts text to lowercase
    - standardize_apostrophe (bool): If True, standardizes apostrophe characters
    - remove_brackets (bool): If True, removes brackets
    - remove_trailing_numbers (bool): If True, removes trailing numbers
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

def process_coda_dataset(csv_path: str, output_path: str):
    """
    Process the CODA CSV file and output a cleaned dictionary
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Rename columns
    df.rename(columns={'Word': 'Keyword', 'Category Types': 'Label'}, inplace=True)
    
    # Fill missing values
    for early_col, new_col in [('Early Quote', 'Quote'), ('Early Word Before', 'Word Before'),
                              ('Early Word After', 'Word After'), ('Early Category Type', 'Label')]:
        df[new_col].fillna(df[early_col], inplace=True)

    # Drop rows with no Keyword and non-Greek Keywords
    pat = '[ء-ي]+'
    df = df.dropna(subset=['Keyword']).copy()
    df = df[~df['Keyword'].str.contains(pat, na=False)]
    
    # Clean data
    df_replacements = {
        r'\d+': '',  # Numbers
        '-': '',     # Hyphens
        ' +': ' ',   # Multiple spaces
    }
    
    keyword_replacements = {
        r'\n': '',   # New line
        ',': '',     # Comma
        r'\.': '',   # Period
        r'\·': '',   # Interpunkt
        r'\s+$': ''  # End punctuation
    }
    
    # Apply cleanings
    columns_to_clean = ['Early Quote', 'Quote', 'Early Word Before', 'Word Before', 
                       'Early Word After', 'Word After', 'Keyword']
    
    for col in columns_to_clean:
        for pattern, replacement in df_replacements.items():
            df[col].replace(pattern, replacement, regex=True, inplace=True)
            
    for pattern, replacement in keyword_replacements.items():
        df['Keyword'].replace(pattern, replacement, regex=True, inplace=True)
        
    # Create the dictionary
    processed_dict = {}
    
    # Process for both normalization forms
    word_lemma_pairs = df.dropna(subset=['Keyword', 'Lemma'])
    
    for form in ('NFKD', 'NFKC'):
        processed_dict[form] = {}
        
        for _, row in word_lemma_pairs.iterrows():
            # Normalize word form (preserve case, standardize apostrophes)
            word = normalize_text(
                text=row['Keyword'],
                form=form,
                remove_accents=False,
                lowercase=False,
                standardize_apostrophe=True,
                remove_brackets=True,
                remove_trailing_numbers=True
            )
            
            # Normalize lemma form (lowercase, standardize apostrophes)
            lemma = normalize_text(
                text=row['Lemma'],
                form=form,
                remove_accents=False,
                lowercase=True,
                standardize_apostrophe=True,
                remove_brackets=True,
                remove_trailing_numbers=True
            )
            
            # Skip empty or invalid entries
            if word and lemma not in ["_", " ", ""]:
                if word not in processed_dict[form]:
                    processed_dict[form][word] = {}
                processed_dict[form][word][lemma] = ["Coda"]
                
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    csv_path = "../assets/NER_assets/Ancient_Words_12_5_22.csv"
    output_path = "../assets/processed_coda_lemmas.json"
    
    process_coda_dataset(csv_path, output_path)