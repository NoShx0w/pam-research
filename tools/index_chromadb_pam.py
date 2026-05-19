#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


INCLUDE_SUFFIXES = {".md", ".py", ".json"}
EXCLUDE_PARTS = {
    ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    "outputs", "node_modules",
}


def should_index(path: Path) -> bool:
    if path.suffix not in INCLUDE_SUFFIXES:
        return False
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.stat().st_size > 300_000:
        return False
    return True


def chunk_text(text: str, max_chars: int = 2400, overlap: int = 300) -> list[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def stable_id(path: Path, chunk_index: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--collection", default="pam_docs_code_v1")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    client = chromadb.HttpClient(host=args.host, port=args.port)

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    embedder = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedder,
        metadata={"project": "pam-research", "kind": "docs_code"},
    )

    ids = []
    docs = []
    metas = []

    files = [p for p in root.rglob("*") if p.is_file() and should_index(p)]
    files.sort()

    for path in files:
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(stable_id(rel, i, chunk))
            docs.append(chunk)
            metas.append(
                {
                    "path": str(rel),
                    "chunk_index": i,
                    "suffix": path.suffix,
                    "bytes": path.stat().st_size,
                }
            )

            if len(ids) >= 100:
                collection.upsert(ids=ids, documents=docs, metadatas=metas)
                ids, docs, metas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)

    print("indexed files:", len(files))
    print("collection count:", collection.count())


if __name__ == "__main__":
    main()
