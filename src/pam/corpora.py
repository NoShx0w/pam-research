from __future__ import annotations

import json
from pathlib import Path


CORPORA_ROOT = Path("observatory/corpora")
REGISTRY_PATH = CORPORA_ROOT / "registry.json"


def load_registry() -> dict[str, str]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Corpus registry must be a JSON object.")
    return {str(k): str(v) for k, v in data.items()}


def load_corpus(name: str) -> list[str]:
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"Unknown corpus '{name}'")
    path = CORPORA_ROOT / registry[name]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Corpus file for '{name}' must contain a JSON list.")
    return [str(x) for x in data]


texts_C = load_corpus("C")
texts_Cp = load_corpus("Cp")
texts_Cp2 = load_corpus("Cp2")
texts_Cp3 = load_corpus("Cp3")
texts_Cp4 = load_corpus("Cp4")


__all__ = [
    "load_registry",
    "load_corpus",
    "texts_C",
    "texts_Cp",
    "texts_Cp2",
    "texts_Cp3",
    "texts_Cp4",
]