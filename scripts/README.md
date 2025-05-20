# Greek Text Preprocessing Pipeline

A modular preprocessing pipeline for Greek text data, supporting lemmatization, morphology/POS tagging, and span categorization tasks.

## Project Structure

```
scripts/
├── __init__.py
├── utils.py              # Core utilities and text normalization
├── dictionary_utils.py   # Dictionary building and management
├── file_io.py           # File operations and temporary directory management
├── preprocess.py        # Common preprocessing routines
├── lemma_preprocess.py  # Lemmatization-specific preprocessing
├── morphology_preprocess.py  # Morphology/POS preprocessing
├── spancat_preprocess.py     # Span categorization preprocessing
├── analysis_utils.py    # Model evaluation and analysis utilities
└── main.py             # Master runner for the pipeline
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:
```bash
pip install spacy pandas tqdm cassis sklearn typing-extensions
```
3. Install the Greek language model:
```bash
python -m spacy download grc_proiel_trf
```

## Usage

### Running the Pipeline

The pipeline can be run in two modes: processing and evaluation.

#### Processing Mode

To run preprocessing steps:

```bash
python -m scripts.main process \
    --input-dir data/raw \
    --output-dir data/processed \
    --steps lemma morphology spancat \
    --model grc_proiel_trf \
    --form NFC \
    --debug
```

Available steps:
- `lemma`: Lemmatization preprocessing
- `morphology`: Morphology/POS preprocessing
- `spancat`: Span categorization preprocessing

#### Evaluation Mode

To evaluate model performance:

```bash
python -m scripts.main evaluate \
    --model-paths path/to/model1 path/to/model2 \
    --model-names "Model 1" "Model 2" \
    --test-data path/to/test.spacy \
    --eval-type lemma \
    --form NFC \
    --eval-pos \
    --eval-morph \
    --output results.csv \
    --debug
```

Evaluation types:
- `lemma`: Evaluate lemmatization (optionally with POS and morphology)
- `ner`: Evaluate named entity recognition
- `spancat`: Evaluate span categorization

### Input Directory Structure

The input directory should be organized as follows:

```
data/raw/
├── lemma/
│   ├── Ancient_Words.csv        # CODA dataset
│   ├── conllu/                  # CONLLU files
│   ├── dendrosearch_lemma_dict.txt
│   └── inception/               # INCEpTION files
├── morphology/
│   └── *.conllu                # CONLLU files with morphology
└── spancat/
    └── *.json                  # Span annotation files
```

### Output Structure

The processed files will be organized as follows:

```
data/processed/
├── lemma/
│   ├── nfc/
│   │   └── processed.spacy
│   └── nfkc/
│       └── processed.spacy
├── morphology/
│   └── morphology.spacy
└── spancat/
    └── spans.spacy
```

## Module Details

### utils.py
- Text normalization utilities
- Unicode normalization handling
- Common constants and configurations

### dictionary_utils.py
- Dictionary building from various sources
- Dictionary combination and processing
- Source weighting for lemma selection

### file_io.py
- File reading/writing operations
- Temporary directory management
- spaCy document saving/loading

### preprocess.py
- Common text preprocessing routines
- Batch processing utilities
- Training data preparation

### lemma_preprocess.py
- Lemmatization-specific processing
- Dictionary-based corrections
- Sentence processing

### morphology_preprocess.py
- CONLLU file processing
- Morphological feature extraction
- POS tag processing

### spancat_preprocess.py
- Span annotation processing
- Span validation and statistics
- Document creation with spans

### analysis_utils.py
- Model evaluation utilities
- Performance metrics calculation
- Result analysis and visualization

### main.py
- Pipeline orchestration
- Command-line interface
- Evaluation framework

## Evaluation Capabilities

### Lemmatization Evaluation
- Token-level accuracy
- Support for multiple models
- Optional POS and morphology evaluation
- Detailed error analysis

### NER Evaluation
- Entity-level precision, recall, F1
- Support for multiple models
- Span boundary analysis
- Label distribution analysis

### Span Categorization Evaluation
- Span-level metrics
- Support for multiple models
- Overlap analysis
- Score threshold analysis

## Development

### Adding New Steps

To add a new preprocessing step:

1. Create a new module (e.g., `new_step_preprocess.py`)
2. Implement the preprocessing logic
3. Add the step to the valid steps in `main.py`
4. Update the input/output directory structure documentation

### Adding New Evaluations

To add a new evaluation type:

1. Add evaluation class to `analysis_utils.py`
2. Implement required metrics and analysis
3. Add evaluation type to `main.py`
4. Update documentation with examples

### Testing

Each module includes error handling and logging. Use the `--debug` flag for detailed logging during development and testing.

## License

This project is licensed under the MIT License - see the LICENSE file for details.