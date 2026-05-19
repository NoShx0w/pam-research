#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import chromadb
from chromadb.utils import embedding_functions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--host", default=os.environ.get("CHROMA_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CHROMA_PORT", "8000")))
    ap.add_argument("--collection", default="pam_docs_code_v1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--chars", type=int, default=900)
    args = ap.parse_args()

    query = " ".join(args.query)

    client = chromadb.HttpClient(host=args.host, port=args.port)
    collection = client.get_collection(
        args.collection,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )

    res = collection.query(query_texts=[query], n_results=args.n)

    print("QUERY:", query)
    print("COLLECTION:", args.collection)

    for i, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]),
        start=1,
    ):
        print("\n" + "=" * 80)
        print(f"{i}. distance={dist:.6g}")
        print(f"path={meta.get('path')} chunk={meta.get('chunk_index')}")
        print("-" * 80)
        print(doc[: args.chars])


if __name__ == "__main__":
    main()
