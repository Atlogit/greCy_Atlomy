import os
import sys
import traceback
import spacy
from spacy.tokens import DocBin
from typing import Dict, List, Union, Set, Any

def find_spancat_files(base_dir: str) -> List[str]:
    """
    Recursively find SpanCat .spacy files in the given directory.
    
    Args:
        base_dir (str): Base directory to search for .spacy files
    
    Returns:
        List of absolute file paths
    """
    spancat_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.spacy') and 'spancat' in file.lower() and 'NFKC' in file:
                spancat_files.append(os.path.abspath(os.path.join(root, file)))
    return spancat_files

def analyze_spancat_docbin(file_path: str, nlp) -> Dict[str, Any]:
    """
    Analyze a SpaCy DocBin file for SpanCat details.
    
    Args:
        file_path (str): Path to the .spacy file
        nlp (spacy.Language): SpaCy language model
    
    Returns:
        dict: Detailed analysis of the SpanCat DocBin file
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None

        # Load the DocBin
        doc_bin = DocBin().from_disk(file_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        # Collect SpanCat information
        total_documents = len(docs)
        total_spans = 0
        total_tokens = 0
        span_categories = {}
        
        for doc in docs:
            # Count tokens
            total_tokens += len(doc)
            
            # Check if the doc has a SpanCat component
            if doc.spans and "sc" in doc.spans:
                for span in doc.spans["sc"]:
                    # Count spans
                    total_spans += 1
                    
                    # Track span categories
                    if span.label_ not in span_categories:
                        span_categories[span.label_] = 0
                    span_categories[span.label_] += 1
        
        return {
            'file_path': file_path,
            'total_documents': total_documents,
            'total_spans': total_spans,
            'total_tokens': total_tokens,
            'span_categories': span_categories
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        traceback.print_exc()
        return None

def categorize_spancat_files(files: List[str]) -> Dict[str, List[str]]:
    """
    Categorize SpanCat files into groups based on dataset type.
    
    Args:
        files (List[str]): List of file paths
    
    Returns:
        Dict of categorized file groups
    """
    categories = {
        'spancat_train_NFKC': [],
        'spancat_dev_NFKC': [],
        'spancat_test_NFKC': [],
        'other': []
    }
    
    for file in files:
        filename = os.path.basename(file)
        
        if 'spancat' in filename.lower() and 'NFKC' in filename:
            if 'train' in filename.lower():
                categories['spancat_train_NFKC'].append(file)
            elif 'dev' in filename.lower():
                categories['spancat_dev_NFKC'].append(file)
            elif 'test' in filename.lower():
                categories['spancat_test_NFKC'].append(file)
            else:
                categories['other'].append(file)
        else:
            categories['other'].append(file)
    
    return categories

def analyze_spancat_comprehensive(nlp) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensively analyze SpanCat files across train, dev, and test folders.
    
    Args:
        nlp (spacy.Language): SpaCy language model
    
    Returns:
        Dict containing detailed SpanCat analysis
    """
    # Find SpanCat files in train, dev, and test directories
    base_dirs = [
        os.path.abspath('corpus/train'),
        os.path.abspath('corpus/dev'),
        os.path.abspath('corpus/test')
    ]
    
    # Collect all .spacy files
    all_files = []
    for base_dir in base_dirs:
        all_files.extend(find_spancat_files(base_dir))
    
    # Categorize files
    file_categories = categorize_spancat_files(all_files)
    
    # Analyze each category
    analysis_results = {}
    for category, files in file_categories.items():
        if not files:
            continue
        
        category_analysis = {
            'total_documents': 0,
            'total_spans': 0,
            'total_tokens': 0,
            'span_categories': {},
            'file_details': []
        }
        
        for file in files:
            file_analysis = analyze_spancat_docbin(file, nlp)
            if file_analysis:
                category_analysis['total_documents'] += file_analysis['total_documents']
                category_analysis['total_spans'] += file_analysis['total_spans']
                category_analysis['total_tokens'] += file_analysis['total_tokens']
                
                # Aggregate span categories
                for cat, count in file_analysis['span_categories'].items():
                    if cat not in category_analysis['span_categories']:
                        category_analysis['span_categories'][cat] = 0
                    category_analysis['span_categories'][cat] += count
                
                category_analysis['file_details'].append({
                    'filename': os.path.basename(file),
                    'total_documents': file_analysis['total_documents'],
                    'total_spans': file_analysis['total_spans'],
                    'total_tokens': file_analysis['total_tokens'],
                    'span_categories': file_analysis['span_categories']
                })
        
        analysis_results[category] = category_analysis
    
    # Calculate total details
    total_analysis = {
        'total_documents': sum(category['total_documents'] for category in analysis_results.values()),
        'total_spans': sum(category['total_spans'] for category in analysis_results.values()),
        'total_tokens': sum(category['total_tokens'] for category in analysis_results.values()),
        'span_categories': {},
    }
    
    # Aggregate span categories across all categories
    for category in analysis_results.values():
        for cat, count in category['span_categories'].items():
            if cat not in total_analysis['span_categories']:
                total_analysis['span_categories'][cat] = 0
            total_analysis['span_categories'][cat] += count
    
    analysis_results['total'] = total_analysis
    
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

    # Perform comprehensive SpanCat analysis
    spancat_analysis = analyze_spancat_comprehensive(nlp)
    
    # Print results
    print("\nSpanCat Analysis Results:")
    for category, data in spancat_analysis.items():
        print(f"\n{category} Group:")
        print(f"  Total Documents: {data['total_documents']}")
        print(f"  Total Spans: {data['total_spans']}")
        print(f"  Total Tokens: {data['total_tokens']}")
        print("  Span Categories:")
        for cat, count in data['span_categories'].items():
            print(f"    {cat}: {count}")
        
        if category != 'total':
            print("\n  File Details:")
            for file_detail in data['file_details']:
                print(f"    {file_detail['filename']}:")
                print(f"      Documents: {file_detail['total_documents']}")
                print(f"      Spans: {file_detail['total_spans']}")
                print(f"      Tokens: {file_detail['total_tokens']}")
                print("      Span Categories:")
                for cat, count in file_detail['span_categories'].items():
                    print(f"        {cat}: {count}")

if __name__ == "__main__":
    main()