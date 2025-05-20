#Lemma evaluator with pos and morph#
 
import pandas as pd
from tqdm import tqdm
import unicodedata as ud
import warnings
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Any

# Define apostrophes and correct_apostrophe as class variables
apostrophes = ["᾽", "᾿", "'", "'", "'"]
correct_apostrophe = "ʼ"

class LemmaEvaluator:
    @staticmethod
    def clean_and_remove_accents(text: str) -> str:
        """
        Cleans the given text by removing diacritics (accents), except for specific characters,
        and converting it to lowercase.
        """
        allowed_characters = [' ̓', "᾿", "᾽", "'", "'", "'", 'ʼ', '̓']  # Including the Greek apostrophe
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        try:
            non_accent_chars = [c for c in ud.normalize('NFKD', text)
            if ud.category(c) != 'Mn' or c in allowed_characters]
            return ''.join(non_accent_chars)
        
        except Exception as e:
            # A more generic exception handling if unexpected errors occur
            print(f"An error occurred: {e}")
            return text
    
    @staticmethod
    def normalize_text(text: str, form: str = 'NFKD',
                      remove_accents: bool = False,
                      lowercase: bool = False,
                      standardize_apostrophe: bool = True,
                      remove_brackets: bool = False,
                      remove_trailing_numbers: bool = False,
                      remove_extra_spaces: bool = False,
                      debug: bool = False) -> str:
        """
        Applies multiple text normalization and cleaning steps on the input text.

        Parameters:
        - text (str): The text to be normalized.
        - form (str): Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD').
        - lowercase (bool): If True, the text is converted to lowercase.
        - standardize_apostrophe (bool): If True, replaces all defined apostrophe characters with a standard one.
        - remove_brackets_only (bool): If True, removes the brackets themselves.
        - remove_trailing_numbers (bool): If True, strips leading or trailing digits from the text.
        
        Returns:
        - str: The processed text.
        """
        normalized_text = text  # Initialize normalized_text with the original text

        # Function to print before and after states for each operation during debugging
        def debug_print(operation_name, before, after):
            if debug:
                print(f"{operation_name} - Before: {before}")
                print(f"{operation_name} - After: {after}")

        # Standardize apostrophe characters if required
        if standardize_apostrophe:
            before_text = normalized_text
            for apos in apostrophes:
                normalized_text = normalized_text.replace(apos, correct_apostrophe)
            debug_print("Standardizing apostrophes", before_text, normalized_text)
            
        if remove_accents:
            before_text = normalized_text
            try:
                normalized_text = LemmaEvaluator.clean_and_remove_accents(normalized_text)
            except Exception as e:
                print(f"An error occurred while removing accents: {e}")
                # Decide what to do here: return the original text, a special value, or stop the process
                return text
            debug_print("Removing accents", before_text, normalized_text)
            
        # Convert to lowercase if required
        if lowercase:
            before_text = normalized_text
            normalized_text = normalized_text.lower()
            debug_print("Lowercase conversion", before_text, normalized_text)

        # Unicode normalization
        if form:
            before_text = normalized_text
            # Handle the form parameter correctly for type checking
            if form == 'NFC':
                normalized_text = ud.normalize('NFC', normalized_text)
            elif form == 'NFD':
                normalized_text = ud.normalize('NFD', normalized_text)
            elif form == 'NFKC':
                normalized_text = ud.normalize('NFKC', normalized_text)
            elif form == 'NFKD':
                normalized_text = ud.normalize('NFKD', normalized_text)
            else:
                # Default to NFKD if somehow an invalid value got through
                normalized_text = ud.normalize('NFKD', normalized_text)
            debug_print("Unicode normalization", before_text, normalized_text)
                
        # Remove brackets only if required
        if remove_brackets:
            before_text = normalized_text
            normalized_text = re.sub(r'[\(\)\[\]]', '', normalized_text)
            debug_print("Removing brackets", before_text, normalized_text)
            
        # Remove trailing numbers if required
        if remove_trailing_numbers:
            before_text = normalized_text
            normalized_text = re.sub(r'^\d+|\d+$', '', normalized_text)
            debug_print("Removing trailing numbers", before_text, normalized_text)

        # Remove multiple spaces and leading/trailing spaces
        if remove_extra_spaces:
            before_text = normalized_text
            normalized_text = ' '.join(normalized_text.split()).strip()
            debug_print("Removing extra spaces", before_text, normalized_text)

        return normalized_text
        
    def __init__(self, models, model_names=None, norm_method='NFKD'):
        self.models = models
        self.model_names = model_names or [f'Model {i+1}' for i in range(len(models))]
        self.norm_method = norm_method

        # Check if normalization method is specified and valid
        if self.norm_method not in ('NFC', 'NFD', 'NFKC', 'NFKD'):
            warnings.warn(f"Invalid normalization method: {self.norm_method}. Using NFKD as default.", UserWarning)
            self.norm_method = 'NFKD'
            
    def normalize_document_text(self, text):
        """
        Normalize the document text using the specified normalization method.
        This ensures consistent encoding throughout the evaluation process.
        """
        # Use the normalize_text method with the specified normalization form
        return self.normalize_text(text, form=self.norm_method, standardize_apostrophe=True)

    def find_token_mapping(self, gold_doc, pred_doc, use_normalization=True):
        """
        Create a mapping between gold and predicted tokens to handle tokenization differences.
        Returns a dictionary mapping gold token indices to predicted token indices.
        
        This enhanced version uses character spans to create more accurate mappings,
        which is especially important for Greek text with different tokenization schemes.
        """
        gold_to_pred = {}
        
        # Get the original text
        original_text = gold_doc.text
        
        # Create character-to-token mappings for both documents
        gold_char_to_token = {}
        for token_idx, token in enumerate(gold_doc):
            for char_idx in range(token.idx, token.idx + len(token.text)):
                gold_char_to_token[char_idx] = token_idx
                
        pred_char_to_token = {}
        for token_idx, token in enumerate(pred_doc):
            for char_idx in range(token.idx, token.idx + len(token.text)):
                pred_char_to_token[char_idx] = token_idx
        
        # Map gold tokens to predicted tokens based on character overlap and normalization
        for gold_idx, gold_token in enumerate(gold_doc):
            # Get character span for this gold token
            start_char = gold_token.idx
            end_char = start_char + len(gold_token.text)
            
            # Find which predicted tokens overlap with this character span
            pred_token_counts = {}
            for char_idx in range(start_char, end_char):
                if char_idx in pred_char_to_token:
                    pred_idx = pred_char_to_token[char_idx]
                    pred_token_counts[pred_idx] = pred_token_counts.get(pred_idx, 0) + 1
            
            # If we found overlapping tokens by character position
            if pred_token_counts:
                best_pred_idx = max(pred_token_counts.items(), key=lambda x: x[1])[0]
                gold_to_pred[gold_idx] = best_pred_idx
                
                # Debug output for significant mismatches
                gold_text = gold_token.text
                pred_text = pred_doc[best_pred_idx].text if best_pred_idx < len(pred_doc) else "N/A"
                if gold_text != pred_text:
                    print(f"Mapped different tokens: Gold[{gold_idx}]='{gold_text}' → Pred[{best_pred_idx}]='{pred_text}'")
            
            # If no overlap found and normalization is enabled, try matching based on normalized text
            elif use_normalization and gold_idx not in gold_to_pred:
                # Normalize the gold token text
                gold_norm = self.normalize_text(gold_token.text, form='NFC', standardize_apostrophe=True, remove_accents=True)
                
                # Try to find a match among predicted tokens
                for pred_idx, pred_token in enumerate(pred_doc):
                    # Skip tokens that are already mapped
                    if pred_idx in gold_to_pred.values():
                        continue
                        
                    # Normalize the predicted token text
                    pred_norm = self.normalize_text(pred_token.text, form='NFC', standardize_apostrophe=True, remove_accents=True)
                    
                    # Check if normalized texts match
                    if gold_norm == pred_norm:
                        gold_to_pred[gold_idx] = pred_idx
                        print(f"Mapped via normalization: Gold[{gold_idx}]='{gold_token.text}' → Pred[{pred_idx}]='{pred_token.text}'")
                        break
        
        # Special handling for apostrophes and other common Greek text issues
        for gold_idx in range(len(gold_doc) - 1):
            if gold_idx not in gold_to_pred and gold_idx + 1 in gold_to_pred:
                # Check if this might be an apostrophe case
                if len(gold_doc[gold_idx].text) == 1 and gold_doc[gold_idx].text in ["'", "ʼ", "'", "᾿", "᾽"]:
                    # Map the apostrophe to the same token as the next token
                    gold_to_pred[gold_idx] = gold_to_pred[gold_idx + 1]
                    print(f"Mapped apostrophe: Gold[{gold_idx}]='{gold_doc[gold_idx].text}' → same as Gold[{gold_idx+1}]")
                
                # Try combining with next token and check if normalized version matches
                elif gold_idx + 1 in gold_to_pred and use_normalization:
                    combined_text = gold_doc[gold_idx].text + gold_doc[gold_idx + 1].text
                    combined_norm = self.normalize_text(combined_text, form='NFC', standardize_apostrophe=True, remove_accents=True)
                    
                    pred_idx = gold_to_pred[gold_idx + 1]
                    if pred_idx < len(pred_doc):
                        pred_norm = self.normalize_text(pred_doc[pred_idx].text, form='NFC', standardize_apostrophe=True, remove_accents=True)
                        
                        if combined_norm == pred_norm:
                            gold_to_pred[gold_idx] = pred_idx
                            print(f"Mapped combined tokens: Gold[{gold_idx}]='{gold_doc[gold_idx].text}' + "
                                  f"Gold[{gold_idx+1}]='{gold_doc[gold_idx+1].text}' → "
                                  f"Pred[{pred_idx}]='{pred_doc[pred_idx].text}'")
        
        return gold_to_pred
        
    def evaluate_lemmas(self, docs, evaluate_lemma=True, evaluate_pos=False, evaluate_tag=False, evaluate_morph=False, use_normalization=True):
        data = []
        skipped_tokens = {
            'total_tokens': 0,
            'empty_gold_lemma': 0,
            'empty_gold_pos': 0,
            'empty_gold_tag': 0,
            'empty_gold_morph': 0,
            'token_length_mismatch': 0
        }

        for doc in tqdm(docs, desc="Processing documents", total=len(docs)):
            # Get the original document text
            doc_text = doc.text
            
            # Normalize the document text before processing
            normalized_text = self.normalize_text(doc_text, form=self.norm_method, standardize_apostrophe=True)
            
            # Process the normalized text with each model
            predicted_docs = [model(normalized_text) for model in self.models]
            
            # Debug: Check tokenization differences
            print("\nTokenization comparison for document:")
            print(f"Original text (first 50 chars): {doc_text[:50]}...")
            print(f"Normalized text (first 50 chars): {normalized_text[:50]}...")
            print(f"Gold doc tokens: {[token.text for token in doc][:10]}...")
            
            # Create token mappings for each model
            token_mappings = []
            for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                print(f"{model_name} tokens: {[token.text for token in predicted_doc][:10]}...")
                
                # Create mapping between gold and predicted tokens
                mapping = self.find_token_mapping(doc, predicted_doc, use_normalization=use_normalization)
                token_mappings.append(mapping)
                
                # Print some mapping examples for debugging
                print(f"Token mapping examples for {model_name}:")
                for gold_idx, pred_idx in list(mapping.items())[:5]:
                    if gold_idx < len(doc) and pred_idx < len(predicted_doc):
                        print(f"  Gold[{gold_idx}]: '{doc[gold_idx].text}' (lemma: '{doc[gold_idx].lemma_}') → "
                              f"Pred[{pred_idx}]: '{predicted_doc[pred_idx].text}' (lemma: '{predicted_doc[pred_idx].lemma_}')")
                
                # Check for unmapped tokens
                unmapped = [idx for idx in range(len(doc)) if idx not in mapping]
                if unmapped:
                    print(f"  Warning: {len(unmapped)} unmapped gold tokens for {model_name}")
                    for idx in unmapped[:3]:  # Show first few unmapped tokens
                        print(f"    Unmapped Gold[{idx}]: '{doc[idx].text}' (lemma: '{doc[idx].lemma_}')")
            
            for token_idx, token in enumerate(doc):
                skipped_tokens['total_tokens'] += 1
                
                # Skip tokens with empty or invalid data based on evaluation flags
                skip_token = False
                
                if evaluate_lemma and (not token.lemma_ or token.lemma_.strip() == ''):
                    skipped_tokens['empty_gold_lemma'] += 1
                    skip_token = True
                
                if evaluate_pos and (not token.pos_ or token.pos_.strip() == ''):
                    skipped_tokens['empty_gold_pos'] += 1
                    skip_token = True
                
                if evaluate_tag and (not token.tag_ or token.tag_.strip() == ''):
                    skipped_tokens['empty_gold_tag'] += 1
                    skip_token = True
                
                if evaluate_morph:
                    morph_dict = token.morph.to_dict()
                    if not morph_dict or all(not value for value in morph_dict.values()):
                        skipped_tokens['empty_gold_morph'] += 1
                        skip_token = True
                
                if skip_token:
                    continue

                token_data = {"Text": doc.text, "Token": token.text}

                if evaluate_lemma:
                    token_data["Gold Lemma"] = token.lemma_
                    for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                        # Get the mapped token index from our mapping
                        mapped_idx = token_mappings[i].get(token_idx)
                        
                        if mapped_idx is not None and mapped_idx < len(predicted_doc):
                            # Use the mapped token
                            token_data[f"{model_name} Lemma"] = predicted_doc[mapped_idx].lemma_
                        elif token_idx < len(predicted_doc):
                            # Fall back to index-based alignment
                            token_data[f"{model_name} Lemma"] = predicted_doc[token_idx].lemma_
                        else:
                            token_data[f"{model_name} Lemma"] = "N/A"
                            skipped_tokens['token_length_mismatch'] += 1

                if evaluate_pos:
                    token_data["Gold POS"] = token.pos_
                    for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                        # Get the mapped token index from our mapping
                        mapped_idx = token_mappings[i].get(token_idx)
                        
                        if mapped_idx is not None and mapped_idx < len(predicted_doc):
                            # Use the mapped token
                            token_data[f"{model_name} POS"] = predicted_doc[mapped_idx].pos_
                        elif token_idx < len(predicted_doc):
                            # Fall back to index-based alignment
                            token_data[f"{model_name} POS"] = predicted_doc[token_idx].pos_
                        else:
                            token_data[f"{model_name} POS"] = "N/A"
                            skipped_tokens['token_length_mismatch'] += 1

                if evaluate_tag:
                    token_data["Gold TAG"] = token.tag_
                    for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                        # Get the mapped token index from our mapping
                        mapped_idx = token_mappings[i].get(token_idx)
                        
                        if mapped_idx is not None and mapped_idx < len(predicted_doc):
                            # Use the mapped token
                            token_data[f"{model_name} TAG"] = predicted_doc[mapped_idx].tag_
                        elif token_idx < len(predicted_doc):
                            # Fall back to index-based alignment
                            token_data[f"{model_name} TAG"] = predicted_doc[token_idx].tag_
                        else:
                            token_data[f"{model_name} TAG"] = "N/A"
                            skipped_tokens['token_length_mismatch'] += 1

                if evaluate_morph:
                    token_data["Gold Morph"] = str(token.morph.to_dict())
                    for i, (model_name, predicted_doc) in enumerate(zip(self.model_names, predicted_docs)):
                        # Get the mapped token index from our mapping
                        mapped_idx = token_mappings[i].get(token_idx)
                        
                        if mapped_idx is not None and mapped_idx < len(predicted_doc):
                            # Use the mapped token
                            token_data[f"{model_name} Morph"] = str(predicted_doc[mapped_idx].morph.to_dict())
                        elif token_idx < len(predicted_doc):
                            # Fall back to index-based alignment
                            token_data[f"{model_name} Morph"] = str(predicted_doc[token_idx].morph.to_dict())
                        else:
                            token_data[f"{model_name} Morph"] = "N/A"
                            skipped_tokens['token_length_mismatch'] += 1

                data.append(token_data)

        # Detailed logging about skipped tokens
        print("\nToken Skipping Statistics:")
        print(f"Total tokens processed: {skipped_tokens['total_tokens']}")
        
        if evaluate_lemma:
            print(f"Tokens skipped due to empty gold lemma: {skipped_tokens['empty_gold_lemma']} ({skipped_tokens['empty_gold_lemma']/skipped_tokens['total_tokens']*100:.2f}%)")
        
        if evaluate_pos:
            print(f"Tokens skipped due to empty gold POS: {skipped_tokens['empty_gold_pos']} ({skipped_tokens['empty_gold_pos']/skipped_tokens['total_tokens']*100:.2f}%)")
        
        if evaluate_tag:
            print(f"Tokens skipped due to empty gold TAG: {skipped_tokens['empty_gold_tag']} ({skipped_tokens['empty_gold_tag']/skipped_tokens['total_tokens']*100:.2f}%)")
        
        if evaluate_morph:
            print(f"Tokens skipped due to empty gold Morphology: {skipped_tokens['empty_gold_morph']} ({skipped_tokens['empty_gold_morph']/skipped_tokens['total_tokens']*100:.2f}%)")
        
        print(f"Tokens skipped due to token length mismatch: {skipped_tokens['token_length_mismatch']} ({skipped_tokens['token_length_mismatch']/skipped_tokens['total_tokens']*100:.2f}%)")

        # Create a DataFrame from the token-level data
        columns = ["Text", "Token"]
        if evaluate_lemma:
            columns += ["Gold Lemma"] + [f"{name} Lemma" for name in self.model_names]
        if evaluate_pos:
            columns += ["Gold POS"] + [f"{name} POS" for name in self.model_names]
        if evaluate_tag:
            columns += ["Gold TAG"] + [f"{name} TAG" for name in self.model_names]
        if evaluate_morph:
            columns += ["Gold Morph"] + [f"{name} Morph" for name in self.model_names]
        
        # Ensure that the number of columns matches the data
        assert len(columns) == len(data[0]), f"{len(columns)} columns specified, but data has {len(data[0])} columns"

        df = pd.DataFrame(data, columns=columns)
        
        # Calculate precision, recall, and F1-score for each model
        metrics = []
        for i in range(len(self.models)):
            model_metrics = [self.model_names[i]]
            print("modl metrics", model_metrics)
            if evaluate_lemma:
                gold_lemmas = df["Gold Lemma"].tolist()
                predicted_lemmas = df[f"{self.model_names[i]} Lemma"].tolist()
                lemma_precision = precision_score(gold_lemmas, predicted_lemmas, average='weighted', zero_division=0)
                lemma_recall = recall_score(gold_lemmas, predicted_lemmas, average='weighted', zero_division=0)
                lemma_f1 = f1_score(gold_lemmas, predicted_lemmas, average='weighted', zero_division=0)
                model_metrics.extend([lemma_precision, lemma_recall, lemma_f1])

            if evaluate_pos:
                gold_pos = df["Gold POS"].tolist()
                predicted_pos = df[f"{self.model_names[i]} POS"].tolist()
                pos_precision = precision_score(gold_pos, predicted_pos, average='weighted', zero_division=0)
                pos_recall = recall_score(gold_pos, predicted_pos, average='weighted', zero_division=0)
                pos_f1 = f1_score(gold_pos, predicted_pos, average='weighted', zero_division=0)
                model_metrics.extend([pos_precision, pos_recall, pos_f1])

            if evaluate_tag:
                gold_tags = df["Gold TAG"].tolist()
                predicted_tags = df[f"{self.model_names[i]} TAG"].tolist()
                tag_precision = precision_score(gold_tags, predicted_tags, average='weighted', zero_division=0)
                tag_recall = recall_score(gold_tags, predicted_tags, average='weighted', zero_division=0)
                tag_f1 = f1_score(gold_tags, predicted_tags, average='weighted', zero_division=0)
                model_metrics.extend([tag_precision, tag_recall, tag_f1])

            if evaluate_morph:
                # Implement evaluation metrics for morphological features if needed
                gold_morph = df["Gold Morph"].tolist()
                predicted_morph = df[f"{self.model_names[i]} Morph"].tolist()
                morph_precision = precision_score(gold_morph, predicted_morph, average='weighted', zero_division=0)
                morph_recall = recall_score(gold_morph, predicted_morph, average='weighted', zero_division=0)
                morph_f1 = f1_score(gold_morph, predicted_morph, average='weighted', zero_division=0)
                model_metrics.extend([morph_precision, morph_recall, morph_f1])
                
            metrics.append(model_metrics)

        # Print out the evaluation metrics in a table format
        columns_metrics = ["Model", "Lemma Precision", "Lemma Recall", "Lemma F1-Score"]
        if evaluate_pos:
            columns_metrics.extend(["POS Precision", "POS Recall", "POS F1-Score"])
        if evaluate_tag:
            columns_metrics.extend(["TAG Precision", "TAG Recall", "TAG F1-Score"])
        if evaluate_morph:
            columns_metrics.extend(["Morph Precision", "Morph Recall", "Morph F1-Score"])
        df_metrics = pd.DataFrame(metrics, columns=columns_metrics)

        print(df_metrics.to_string(index=False))

        return df