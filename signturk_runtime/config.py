"""Paths, configuration and vocabulary of the 226-class ensemble."""

from __future__ import annotations

import json
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1] / "research" / "model_226"


def load_config(root: Path = MODEL_ROOT) -> dict:
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def load_vocabulary(root: Path = MODEL_ROOT) -> list:
    """One entry per model output: {"autsl_id": int, "gloss": str, "en": str}.

    `labels/label_map.json` maps an AUTSL class id to a model index, and
    `labels/class_display_names.json` gives the AUTSL gloss with spaces;
    the gloss used everywhere else in the project has underscores.
    `labels/class_names_en.json` has the English names from AUTSL's
    SignList_ClassId_TR_EN.csv.
    """
    label_map = json.loads((root / "labels" / "label_map.json").read_text(encoding="utf-8"))
    names = json.loads((root / "labels" / "class_display_names.json").read_text(encoding="utf-8"))
    english = json.loads((root / "labels" / "class_names_en.json").read_text(encoding="utf-8"))
    vocabulary = [None] * len(label_map)
    for autsl_id, index in label_map.items():
        vocabulary[int(index)] = {"autsl_id": int(autsl_id), "gloss": names[autsl_id].strip().replace(" ", "_"),
                                  "en": english.get(autsl_id)}
    return vocabulary


def missing_files(config: dict, root: Path = MODEL_ROOT) -> list:
    """Model and extractor files listed in the config that are not on disk."""
    paths = [spec["file"] for spec in config["streams"].values()]
    paths += [config["extractor"]["detector"], config["extractor"]["pose"]]
    return [path for path in paths if not (root / path).exists()]
