# Greek Lemmatizer

A Python script for processing Ancient Greek texts with advanced lemmatization capabilities. This tool combines multiple lemmatization sources and applies sophisticated text normalization techniques.

## Features

- Multiple dictionary sources integration (CODA, Dendrosearch, CONLLU, INCEpTION)
- Advanced text normalization with Unicode support
- Customizable preprocessing options
- Source tracking for lemmatization corrections
- Train/dev/test split functionality
- Comprehensive debugging and logging
- SpaCy document processing and saving

## Prerequisites

```bash
pip install spacy pandas tqdm cassis
python -m spacy download grc_proiel_trf
```

## Directory Structure

```
.
├── assets/
│   ├── NER_assets/
│   │   ├── Ancient_Words_12_5_22.csv
│   │   └── INCEpTION_files/
│   ├── Lemmatization_training_files/
│   │   └── Processed/
│   └── dendrosearch_lemma_dict.txt
├── corpus/
│   ├── train/
│   ├── dev/
│   └── test/
└── src/
    └── greek_lemmatizer.py
```

## Usage

Basic usage:
```bash
python greek_lemmatizer.py \
  --input-dir ../assets/input \
  --output-dir ../corpus \
  --model grc_proiel_trf \
  --forms NFC  # Default normalization form
```

With all options:
```bash
python greek_lemmatizer.py \
  --input-dir ../assets/input \
  --output-dir ../corpus \
  --model grc_proiel_trf \
  --forms NFC NFKC NFKD \  # Process with multiple normalization forms
  --save-corrections corrections.json \
  --batch-size 1000 \
  --debug
```

### Command Line Arguments

- `--input-dir`: Input directory containing files to process (required)
- `--output-dir`: Output directory for processed files (required)
- `--model`: SpaCy model to use (default: "grc_proiel_trf")
- `--forms`: Unicode normalization forms to use (choices: NFC, NFKC, NFD, NFKD; default: NFC)
- `--batch-size`: Number of sentences to process in each batch (default: 1000)
- `--save-corrections`: Path to save corrections JSON (optional)
- `--debug`: Enable debug logging (optional)

### Output Files

For each normalization form specified with `--forms`, the script generates:
1. Train/dev/test splits with form-specific suffixes (e.g., `train_NFC.spacy`, `train_NFKD.spacy`)
2. Separate corrections files if `--save-corrections` is specified:
   - For input `corrections.json` and forms `NFC NFKD`
   - Creates `corrections_NFC.json` and `corrections_NFKD.json`

## Text Processing Features

### Unicode Normalization
- Supports multiple normalization forms (NFC, NFKC, NFD, NFKD)
- Processes each form independently with separate outputs
- Standardizes apostrophes and special characters
- Preserves important diacritical marks

### Text Cleaning
- Accent removal (optional)
- Case normalization
- Bracket removal
- Trailing number removal
- Extra space removal

### Lemma Selection
- Combines lemmas from multiple sources using a weighted scoring system
- Source prioritization weights:
  * INCEpTION (4.0): Highest weight for manually annotated data
  * CODA (3.0): High weight for manually curated dataset
  * CONLLU (2.0): Medium weight for Universal Dependencies format
  * Dendrosearch (1.0): Base weight for automated extraction
- Scoring method:
  * Base score = frequency of lemma across sources
  * Source score = sum of weights of sources providing the lemma
  * Final score = base score × source score
  * Example: A lemma appearing twice in INCEpTION (2 × 4.0 = 8.0) ranks higher than one appearing three times in Dendrosearch (3 × 1.0 = 3.0)
- Tie-breaking strategy:
  * When multiple lemmas have equal scores, combines source information
  * Uses lexicographical ordering for final resolution
  * Example: If "λέγω" and "λέγομαι" have the same score, "λέγω" is chosen
- Handles conflicting lemmatizations by preferring more reliable sources

### Lemmatization Sources
1. CODA Dictionary
   - Source: Ancient_Words_12_5_22.csv
   - Contains manually curated lemmatizations

2. Dendrosearch Dictionary
   - Source: dendrosearch_lemma_dict.txt
   - Specialized Greek lexicon

3. CONLLU Files
   - Source: Lemmatization_training_files/Processed/
   - Universal Dependencies format

4. INCEpTION Files
   - Source: INCEpTION_files/
   - Contains annotated texts with lemma information

## Output

The script produces:

1. Processed SpaCy documents split into:
   - train/lemma_train/
   - dev/lemma_dev/
   - test/lemma_test/

2. Optional corrections JSON file containing:
   - Original sentence
   - Source information
   - Token information
   - Original and corrected lemmas
   - Correction sources

## Debugging

When running with `--debug`, the script provides detailed logging for:
- Text normalization steps
- Lemma selection process
- Dictionary processing
- Sentence processing
- Lemmatization corrections
- Document processing and saving

## Error Handling

The script includes comprehensive error handling for:
- File operations
- Text processing
- Dictionary building
- Document processing
- SpaCy operations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
