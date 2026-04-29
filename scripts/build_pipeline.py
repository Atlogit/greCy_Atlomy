#!/usr/bin/env python3
"""
Build the published Atlomy spaCy pipeline by combining the spancat from one
trained pipeline (with its transformer baked into the spancat via
``replace_listeners``) with the morphologizer / tagger / parser /
trainable_lemmatizer / attribute_ruler from a second, separately trained
pipeline that has working morphology.

Why this shape: the published spancat was trained on top of an upstream
pipeline whose morphologizer was trained on POS-only labels (no Case /
Gender / Number / etc.), so its morph output is empty at inference. The
later wantuta-based lemmatizer pipeline has a properly trained
morphologizer/tagger/parser but no spancat. Naively sourcing the spancat
with ``nlp.add_pipe("spancat", source=other)`` produces garbage because
the spancat's TransformerListener is coupled to the source pipeline's
fine-tuned transformer weights, not the destination's. ``replace_listeners``
bakes the source transformer into the spancat so it travels with it.

Usage::

    python -m scripts.build_pipeline \
        --spancat-source path/to/spancat-pipeline_230524_sweep \
        --lemma-source path/to/lemmatization-trf_25_mar_25_wantuta \
        --output path/to/grc_atlomy_spancat \
        --name grc_atlomy_spancat \
        --version 0.1.0
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import spacy

logger = logging.getLogger(__name__)


def build(
    spancat_source: Path,
    lemma_source: Path,
    output: Path,
    pipeline_name: str = "atlomy_spancat",
    version: str = "0.1.0",
    description: str = (
        "Ancient Greek spaCy pipeline with span categorization "
        "for medical/anatomical terminology."
    ),
    author: str = "",
    email: str = "",
    url: str = "",
    license_str: str = "MIT",
) -> Path:
    """Build the combined pipeline and write it to ``output``."""
    output = output.resolve()
    if output.exists():
        logger.warning("Output exists, overwriting: %s", output)
        shutil.rmtree(output)

    logger.info("Loading spancat source: %s", spancat_source)
    nlp_spancat = spacy.load(spancat_source)
    if "spancat" not in nlp_spancat.pipe_names:
        raise SystemExit(
            f"Source pipeline {spancat_source} has no 'spancat' component "
            f"(pipes: {nlp_spancat.pipe_names})"
        )

    # Bake the listener: replace the spancat's TransformerListener with a
    # frozen copy of the source pipeline's transformer. This makes the
    # spancat self-contained; it carries its own transformer weights and
    # no longer depends on whichever transformer is upstream in the host
    # pipeline.
    logger.info("Baking listener: replace_listeners('transformer', 'spancat', ['model.tok2vec'])")
    nlp_spancat.replace_listeners("transformer", "spancat", ["model.tok2vec"])

    logger.info("Loading host pipeline (provides transformer/morph/tag/parse/lemma): %s", lemma_source)
    nlp = spacy.load(lemma_source)

    if "spancat" in nlp.pipe_names:
        logger.info("Host pipeline already has a 'spancat'; removing it")
        nlp.remove_pipe("spancat")

    logger.info("Sourcing spancat onto host pipeline (last in pipeline)")
    nlp.add_pipe("spancat", source=nlp_spancat, name="spancat", last=True)

    # Update meta.json before serializing.
    nlp.meta["lang"] = "grc"
    nlp.meta["name"] = pipeline_name
    nlp.meta["version"] = version
    nlp.meta["description"] = description
    nlp.meta["author"] = author
    nlp.meta["email"] = email
    nlp.meta["url"] = url
    nlp.meta["license"] = license_str

    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing pipeline to: %s", output)
    nlp.to_disk(output)

    # Sanity: write a small build manifest next to the model
    manifest = {
        "pipeline_name": pipeline_name,
        "version": version,
        "spancat_source": str(spancat_source),
        "lemma_source": str(lemma_source),
        "spacy_version": spacy.__version__,
        "pipe_names": list(nlp.pipe_names),
        "spancat_labels": list(nlp.get_pipe("spancat").labels),
        "spancat_threshold": nlp.get_pipe("spancat").cfg.get("threshold"),
    }
    (output / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    logger.info("Done. pipe_names: %s", nlp.pipe_names)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spancat-source",
        type=Path,
        required=True,
        help="Path to trained pipeline that contains the spancat to publish",
    )
    parser.add_argument(
        "--lemma-source",
        type=Path,
        required=True,
        help="Path to trained pipeline whose morph/tag/parse/lemma we want",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output pipeline directory"
    )
    parser.add_argument(
        "--name",
        default="atlomy_spancat",
        help="Pipeline name (combined with lang gives e.g. grc_atlomy_spancat)",
    )
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--author", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--license", dest="license_str", default="MIT")
    parser.add_argument(
        "--description",
        default=(
            "Ancient Greek spaCy pipeline with span categorization "
            "for medical/anatomical terminology."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # spaCy pulls some things from HF on first load if not already cached;
    # for production builds we want offline-only behavior.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    out = build(
        spancat_source=args.spancat_source,
        lemma_source=args.lemma_source,
        output=args.output,
        pipeline_name=args.name,
        version=args.version,
        description=args.description,
        author=args.author,
        email=args.email,
        url=args.url,
        license_str=args.license_str,
    )
    print(out)


if __name__ == "__main__":
    main()
