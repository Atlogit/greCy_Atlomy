# Legacy notebooks

These are the original development notebooks from the dev_Atlomy branch,
preserved here for historical reference. They are **not** part of the
publishing path and many have hard-coded local paths and stale references
that will not run for a new user.

For a clean walkthrough that works with the published pipeline, use
[`notebooks/demo.ipynb`](../demo.ipynb).

The functionality each notebook prototyped has been migrated to:

| Notebook                              | Replaced by                                                  |
|---------------------------------------|--------------------------------------------------------------|
| `Preprocess.ipynb`                    | `scripts/preprocess.py`, `scripts/conllu_utils.py`           |
| `Greek_Lemmatizer.ipynb`              | `scripts/lemma_preprocess.py`, `scripts/dictionary_utils.py` |
| `Greek_POS_conllu.ipynb`              | `scripts/morphology_preprocess.py`                           |
| `Greek_NER_SpanCat.ipynb`             | `scripts/spancat_preprocess.py`                              |
| `Greek_NER.ipynb`                     | superseded by SpanCat (above)                                |
| `Models Evaluations.ipynb`            | `scripts/main.py evaluate ...`, `scripts/analysis_utils.py`  |
| `Models Evaluations Template.ipynb`   | superseded by `notebooks/demo.ipynb`                         |
| `NER_train.ipynb`                     | spaCy CLI: `python -m spacy train …` (see `project.yml`)     |
| `assemble_training.ipynb`             | spaCy CLI: `python -m spacy train` / `package`               |

Note that `scripts/build_pipeline.py` is what produces the published pipeline
from already-trained component checkpoints; the training itself runs through
the standard spaCy CLI against the configs in `configs/`.
