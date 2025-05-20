import os
import sys
import traceback
import spacy
from spacy.tokens import DocBin
from typing import Dict, List, Union, Set

def find_lemma_files(base_dir: str) -> List[str]:
    """
    Recursively find all .spacy files in the given directory.
    
    Args:
        base_dir (str): Base directory to search for .spacy files
    
    Returns:
        List of absolute file paths
    """
    lemma_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.spacy'):
                lemma_files.append(os.path.abspath(os.path.join(root, file)))
    return lemma_files

def analyze_spacy_docbin(file_path: str, nlp) -> Dict:
    """
    Analyze a SpaCy DocBin file to extract lemma and token information.
    
    Args:
        file_path (str): Path to the .spacy file
        nlp (spacy.Language): SpaCy language model
    
    Returns:
        dict: Analysis of the DocBin file
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None

        # Load the DocBin
        doc_bin = DocBin().from_disk(file_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        # Collect lemma and token information
        total_tokens = 0
        lemmas = set()
        all_lemmas = []
        total_documents = len(docs)
        
        for doc in docs:
            total_tokens += len(doc)
            doc_lemmas = [token.lemma_ for token in doc]
            lemmas.update(doc_lemmas)
            all_lemmas.extend(doc_lemmas)
        
        return {
            'file_path': file_path,
            'total_tokens': total_tokens,
            'total_documents': total_documents,
            'unique_lemmas': lemmas,
            'total_lemmas': all_lemmas
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        traceback.print_exc()
        return None

def categorize_files(files: List[str]) -> Dict[str, List[str]]:
    """
    Categorize files into groups based on normalization and dataset type.
    
    Args:
        files (List[str]): List of file paths
    
    Returns:
        Dict of categorized file groups
    """
    categories = {
        'atlomy_train_NFKC': [],
        'atlomy_train_NFKD': [],
        'atlomy_dev_NFKC': [],
        'atlomy_dev_NFKD': [],
        'atlomy_test_NFKC': [],
        'atlomy_test_NFKD': [],
        'other': []
    }
    
    for file in files:
        filename = os.path.basename(file)
        
        # Check if the file is part of the Atlomy project
        if 'atlomy' in filename.lower():
            if 'train' in filename.lower():
                if 'NFKC' in filename:
                    categories['atlomy_train_NFKC'].append(file)
                elif 'NFKD' in filename:
                    categories['atlomy_train_NFKD'].append(file)
                else:
                    categories['other'].append(file)
            
            elif 'dev' in filename.lower():
                if 'NFKC' in filename:
                    categories['atlomy_dev_NFKC'].append(file)
                elif 'NFKD' in filename:
                    categories['atlomy_dev_NFKD'].append(file)
                else:
                    categories['other'].append(file)
            
            elif 'test' in filename.lower():
                if 'NFKC' in filename:
                    categories['atlomy_test_NFKC'].append(file)
                elif 'NFKD' in filename:
                    categories['atlomy_test_NFKD'].append(file)
                else:
                    categories['other'].append(file)
            
            else:
                categories['other'].append(file)
        else:
            categories['other'].append(file)
    
    return categories

def analyze_lemmas_comprehensive(nlp) -> Dict[str, Dict[str, Union[int, List[Dict]]]]:
    """
    Comprehensively analyze lemmas across train, dev, and test folders.
    
    Args:
        nlp (spacy.Language): SpaCy language model
    
    Returns:
        Dict containing detailed lemma analysis
    """
    # Find lemma files in train, dev, and test directories
    base_dirs = [
        os.path.abspath('corpus/train/lemma_train'),
        os.path.abspath('corpus/dev/lemma_dev'),
        os.path.abspath('corpus/test/lemma_test')
    ]
    
    # Collect all .spacy files
    all_files = []
    for base_dir in base_dirs:
        all_files.extend(find_lemma_files(base_dir))
    
    # Categorize files
    file_categories = categorize_files(all_files)
    
    # Analyze each category
    analysis_results = {}
    for category, files in file_categories.items():
        if not files:
            continue
        
        category_analysis = {
            'total_tokens': 0,
            'total_documents': 0,
            'total_unique_lemmas': set(),
            'total_lemmas': [],
            'file_details': []
        }
        
        for file in files:
            file_analysis = analyze_spacy_docbin(file, nlp)
            if file_analysis:
                category_analysis['total_tokens'] += file_analysis['total_tokens']
                category_analysis['total_documents'] += file_analysis['total_documents']
                category_analysis['total_unique_lemmas'].update(file_analysis['unique_lemmas'])
                category_analysis['total_lemmas'].extend(file_analysis['total_lemmas'])
                category_analysis['file_details'].append({
                    'filename': os.path.basename(file),
                    'total_tokens': file_analysis['total_tokens'],
                    'total_documents': file_analysis['total_documents'],
                    'unique_lemmas': len(file_analysis['unique_lemmas']),
                    'total_lemmas': len(file_analysis['total_lemmas'])
                })
        
        # Convert unique lemmas to count
        category_analysis['total_unique_lemmas'] = len(category_analysis['total_unique_lemmas'])
        category_analysis['total_lemmas'] = len(category_analysis['total_lemmas'])
        
        analysis_results[category] = category_analysis
    
    return analysis_results

def main():
    # Print current working directory and file paths for debugging
    print("Current Working Directory:", os.getcwd())
    
    # Attempt to load the SpaCy model with full path
    model_path = os.path.abspath("training/ATLOMY_G_NER_pipeline/atlomy_full_pipeline_annotation_041224/model-best")
    print("Model Path:", model_path)
    
    try:
        nlp = spacy.load(model_path)
    except Exception as e:
        print(f"Error loading SpaCy model: {e}")
        traceback.print_exc()
        return

    # Perform comprehensive lemma analysis
    lemma_analysis = analyze_lemmas_comprehensive(nlp)
    
    # Print results
    print("\nLemma Analysis Results:")
    for category, data in lemma_analysis.items():
        print(f"\n{category} Group:")
        print(f"  Total Tokens: {data['total_tokens']}")
        print(f"  Total Documents: {data['total_documents']}")
        print(f"  Total Unique Lemmas: {data['total_unique_lemmas']}")
        print(f"  Total Lemmas: {data['total_lemmas']}")
        
        print("\n  File Details:")
        for file_detail in data['file_details']:
            print(f"    {file_detail['filename']}:")
            print(f"      Tokens: {file_detail['total_tokens']}")
            print(f"      Documents: {file_detail['total_documents']}")
            print(f"      Unique Lemmas: {file_detail['unique_lemmas']}")
            print(f"      Total Lemmas: {file_detail['total_lemmas']}")

if __name__ == "__main__":
    main()