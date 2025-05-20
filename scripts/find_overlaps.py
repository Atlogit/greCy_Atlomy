import spacy
from spacy.tokens import DocBin
from pathlib import Path

def load_docs_from_file(file_path, nlp):
    """Load docs from a spacy file and return their texts."""
    doc_bin = DocBin().from_disk(file_path)
    return {doc.text for doc in doc_bin.get_docs(nlp.vocab)}

def main():
    # Load spacy model
    nlp = spacy.load("grc_proiel_trf")
    nlp.disable_pipes(["morphologizer", "tagger", "parser", "attribute_ruler"])

    # Load lemma_train.spacy and lemma_dev.spacy
    train_path = Path("corpus/train/lemma_train/lemma_train.spacy")
    dev_path = Path("corpus/dev/lemma_dev/lemma_dev.spacy")

    print("Loading train texts...")
    train_texts = load_docs_from_file(train_path, nlp)
    print(f"Loaded {len(train_texts)} training texts")

    print("\nLoading dev texts...")
    dev_texts = load_docs_from_file(dev_path, nlp)
    print(f"Loaded {len(dev_texts)} dev texts")

    # Find overlaps
    overlaps = train_texts.intersection(dev_texts)
    print(f"\nFound {len(overlaps)} overlapping texts")

    # Save overlaps to file
    with open("overlapping_texts.txt", "w", encoding="utf-8") as f:
        for text in overlaps:
            f.write(text + "\n\n===\n\n")

    print(f"\nSaved {len(overlaps)} overlapping texts to overlapping_texts.txt")

    # Print some examples
    print("\nExample overlapping texts:")
    for text in list(overlaps)[:3]:  # Show first 3 examples
        print("\n---")
        print(text[:200] + "..." if len(text) > 200 else text)
        print("---")

if __name__ == "__main__":
    main()
