# greCy_Atlomy

`grc_atlomy_spancat` is a [spaCy](https://spacy.io) pipeline for Ancient Greek
that adds **span categorisation of medical and anatomical terminology** on top
of a working morpho-syntactic backbone (transformer + tagger + morphologizer +
parser + lemmatiser).

It is built on the [greCy](https://github.com/jmyerston/greCy) project and the
[Perseus](https://universaldependencies.org/treebanks/grc_perseus/index.html) /
[PROIEL](https://universaldependencies.org/treebanks/grc_proiel/index.html) UD
treebanks; the spancat layer is trained on a hand-annotated corpus from the
Atlomy project (Galen, Hippocrates, etc.).

## Span categories

The spancat predicts spans under `doc.spans["sc"]` with these labels:

| Label                   | Coverage on test set | F1   |
|-------------------------|----------------------|------|
| Body Part               | 161 spans            | 0.93 |
| Topography              | 125 spans            | 0.95 |
| Adjectives/Qualities    |  71 spans            | 0.89 |
| Action Verbs            |  15 spans            | 0.82 |
| Medical                 |  11 spans            | 0.74 |
| Physiology              |   9 spans            | 0.94 |
| Technical Appellation   |   5 spans            | 0.67 |
| Division                |   5 spans            | 0.57 |
| Symmetry/Opposition     |   2 spans            | 0.00 |
| **Overall**             | **404 spans**        | **0.913** |

(Pathology is in the label set but not represented in the test split.)

## Install

```bash
pip install -r requirements.txt
pip install grc_atlomy_spancat-0.1.0.tar.gz
```

The release tarball is attached to GitHub
[releases](https://github.com/Atlogit/greCy_Atlomy/releases). The pipeline is
~1.1 GB because it ships the embedded transformer
(`wantuta/roberta_ancient_greek_mlm`) weights — there is no network fetch on
first load.

## Use

```python
import spacy
nlp = spacy.load("grc_atlomy_spancat")

text = (
    "πρὸς δὲ τὸν καυλὸν τὸν τῆς κύστεως συνήρτηται τὸ αἰδοῖον, "
    "τὸ μὲν ἐξωτάτω τρῆμα συνερρωγὸς εἰς τὸ αὐτό"
)
doc = nlp(text)

for token in doc:
    print(f"{token.text}\t{token.lemma_}\t{token.pos_}\t{token.morph}")

for span in doc.spans["sc"]:
    print(f"[{span.start_char}:{span.end_char}] {span.label_:25s} {span.text}")
```

A more complete walkthrough — including span and dependency-tree rendering
with displaCy — is in [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

## What's in the pipeline

```python
>>> nlp.pipe_names
['transformer', 'morphologizer', 'tagger', 'parser',
 'trainable_lemmatizer', 'attribute_ruler', 'spancat']
```

| Component             | Source       | Notes |
|-----------------------|--------------|-------|
| `transformer`         | wantuta      | RoBERTa-base (768d) trained on Ancient Greek MLM |
| `morphologizer`       | wantuta-fine | Universal Dependencies POS + features |
| `tagger`              | wantuta-fine | PROIEL XPOS tagset (`A-`, `Df`, `Nb`, …) |
| `parser`              | wantuta-fine | UD-style dependencies (UAS ≈ 0.59 on Atlomy in-domain) |
| `trainable_lemmatizer`| wantuta-fine | 95% lemma accuracy on Atlomy CoDA test set |
| `spancat`             | atlomy       | 91% F1; uses a baked-in transformer copy (see below) |

The `spancat` component carries its **own** copy of the transformer weights
because its `TransformerListener` was trained against a different fine-tuning
of the base than the rest of the pipeline. Bolting the spancat onto a
different upstream transformer produces unusable output — see "Building from
source" below for details.

## Building from source

If you want to retrain or rebuild the combined pipeline from your own
component checkpoints, use `scripts/build_pipeline.py`:

```bash
python -m scripts.build_pipeline \
  --spancat-source path/to/your-spancat-pipeline/model-best \
  --lemma-source   path/to/your-lemmatizer-pipeline/model-best \
  --output         out/grc_atlomy_spancat \
  --name atlomy_spancat \
  --version 0.1.0
```

The script does one important thing: it calls
`nlp.replace_listeners("transformer", "spancat", ["model.tok2vec"])` on the
spancat-source pipeline before sourcing the spancat onto the lemma-source.
This bakes the spancat's `TransformerListener` into a self-contained
`Tok2VecTransformer`. Without it, the spancat reads embeddings from the
destination pipeline's transformer — which has been fine-tuned for a
different objective — and produces garbage.

To turn the resulting directory into a pip-installable artifact:

```bash
mkdir -p dist
python -m spacy package out/grc_atlomy_spancat dist/ --build sdist
```

## Training data

The data used to train the components in this pipeline includes:

- **POS / morph / parser / lemma**: PROIEL + Perseus UD treebanks, plus a
  custom Atlomy lemma corpus derived in part from
  [Giuseppe Celano's lemmatised corpus](https://github.com/gcelano/LemmatizedAncientGreekXML).
  Gold `.spacy` files for each split are checked into `corpus/`.
- **SpanCat**: hand-annotated medical/anatomical text from the Atlomy project
  (Galen, Hippocrates, Aristotle), distributed across
  `corpus/{train,dev,test}/spancat_*`.

Annotation work used [INCEpTION](https://inception-project.github.io/) and
an internal CoDA spreadsheet; the original raw sources are not
redistributable, but the prepared `.spacy` corpora used for training are.

## Repository layout

```
greCy_Atlomy/
├── README.md
├── requirements.txt
├── project.yml          # spaCy project file
├── configs/             # spaCy training configs
├── corpus/              # Prepared .spacy training/dev/test data
├── data/                # Labels + augmenter patterns
├── scripts/             # CLI tools and library code
│   ├── build_pipeline.py    # Combines spancat + lemma pipelines
│   ├── main.py              # Preprocess / evaluate CLI
│   ├── analysis_utils.py    # Lemma / NER / SpanCat evaluators
│   └── ...
└── notebooks/
    └── demo.ipynb       # Inference walkthrough
```

## Caveats

- **POS vs tag**: `token.pos_` is Universal POS; `token.tag_` is the PROIEL
  XPOS (`A-`, `Df`, `Nb`, …).
- **Dependency labels**: the parser was trained on a slightly different
  label vocabulary than the Atlomy CoDA evaluation set, so labelled
  attachment scores (LAS) are misleadingly low on that out-of-distribution
  test. UAS (head attachment) is meaningful; LAS is only meaningful against
  UD-PROIEL test data.
- **Tokenisation**: the spaCy tokenizer for `grc` handles most Ancient Greek
  punctuation and apostrophes, but a small fraction of edge cases —
  especially around the modifier letter apostrophe `ʼ` (U+02BC) — tokenise
  differently from some hand-curated gold sets.

## License

MIT — see [LICENSE](LICENSE). Built on:

- [greCy](https://github.com/jmyerston/greCy) (MIT, Jacobo Myerston)
- [PROIEL](https://github.com/proiel/proiel-treebank) and
  [Perseus](https://github.com/PerseusDL/treebank_data) UD treebanks
- [aristoBERTo](https://huggingface.co/Jacobo/aristoBERTo) and
  [`wantuta/roberta_ancient_greek_mlm`](https://huggingface.co/wantuta/roberta_ancient_greek_mlm)

## Citation

If you use this model in academic work, please cite both the upstream greCy
project and the Atlomy span-categorisation work. A `CITATION.cff` is provided.
