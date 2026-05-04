#!/usr/bin/env python3
from __future__ import annotations


"""
Audit corpus response hygiene across PAM corpus variants.

Purpose
-------
Detect simple hygiene phenomena in response texts, focusing on:
- presence / absence of links
- empty responses
- very short responses
- partial / truncated-like responses

This is a lightweight first-pass audit instrument, intended to identify
possible corpus confounds before deeper observatory interpretation.

Expected input
--------------
A directory containing corpus JSON files such as:

    observatory/corpora/
      C.json
      Cp.json
      Cp2.json
      Cp3.json
      Cp4.json

Each file is expected to contain either:
- a JSON list of strings
- a JSON list of dicts with a text-bearing field
- a dict with a top-level list under common keys like "texts", "responses", or "items"

Outputs
-------
<outdir>/
  corpus_response_hygiene_rows.csv
  corpus_response_hygiene_summary.csv
  corpus_response_hygiene_summary.txt

Heuristics
----------
Link detection:
- URL-like regex match on http://, https://, or www.

Empty:
- stripped text length == 0

Very short:
- character length below threshold OR token count below threshold

Partial:
- ends without terminal punctuation
- or ends in trailing connector / punctuation suggesting continuation
- or has unmatched opening brackets / quotes

Truncated-like:
- partial plus stronger abrupt-cut patterns
- ends with unfinished short token / connector / colon / dash
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import pandas as pd


URL_RE = re.compile(
    r"""(?ix)
    \b(
        https?://[^\s<>()\[\]{}"']+
        |
        www\.[^\s<>()\[\]{}"']+
    )
    """
)

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
TERMINAL_RE = re.compile(r'[.!?)"\]»”]\s*$')
TRAILING_CONTINUATION_RE = re.compile(
    r"""(?ix)
    (
        [,;:]\s*$
        |
        [\-–—]\s*$
        |
        \b(and|or|but|because|that|which|who|when|where|while|with|without|if|then|than|to|of|for|in|on|at|from|by|as)\s*$
    )
    """
)


@dataclass(frozen=True)
class Config:
    corpora_root: str = "observatory/corpora"
    outdir: str = "outputs/corpus_response_hygiene_audit"
    very_short_char_threshold: int = 40
    very_short_token_threshold: int = 8
    show_examples_per_corpus: int = 5


def safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_candidate_texts(obj: Any) -> Iterable[tuple[int, str]]:
    """
    Yield (index, text) from a few common corpus JSON shapes.
    """
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            text = extract_text(item)
            yield i, text
        return

    if isinstance(obj, dict):
        for key in ["texts", "responses", "items", "data", "records"]:
            if key in obj and isinstance(obj[key], list):
                for i, item in enumerate(obj[key]):
                    text = extract_text(item)
                    yield i, text
                return

    raise ValueError("Unsupported corpus JSON structure")


def extract_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in [
            "text",
            "response",
            "content",
            "output",
            "assistant_response",
            "message",
            "body",
        ]:
            if key in item:
                value = item[key]
                return "" if value is None else str(value)
    return str(item)


def count_links(text: str) -> int:
    return len(URL_RE.findall(text))


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def unmatched_brackets_or_quotes(text: str) -> bool:
    pairs = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
    ]
    for left, right in pairs:
        if text.count(left) > text.count(right):
            return True

    for quote in ['"', "'", "“", "”", "‘", "’"]:
        # crude but useful first-pass heuristic
        if text.count(quote) % 2 == 1:
            return True

    return False


def classify_partial(text: str) -> tuple[bool, bool]:
    stripped = text.strip()
    if not stripped:
        return False, False

    ends_cleanly = bool(TERMINAL_RE.search(stripped))
    continuation_tail = bool(TRAILING_CONTINUATION_RE.search(stripped))
    unmatched = unmatched_brackets_or_quotes(stripped)

    is_partial = (not ends_cleanly) or continuation_tail or unmatched

    # stronger abrupt-cut signal
    last_token = ""
    toks = TOKEN_RE.findall(stripped)
    if toks:
        last_token = toks[-1]

    truncated_like = False
    if continuation_tail or unmatched:
        truncated_like = True
    elif not ends_cleanly:
        if len(last_token) <= 3:
            truncated_like = True
        elif re.search(r"[a-zA-Z0-9]$", stripped):
            truncated_like = True

    return is_partial, truncated_like


def audit_text(
    text: str,
    *,
    very_short_char_threshold: int,
    very_short_token_threshold: int,
) -> dict[str, object]:
    raw = "" if text is None else str(text)
    stripped = raw.strip()

    n_chars = len(stripped)
    n_tokens = token_count(stripped)
    n_links = count_links(stripped)

    is_empty = n_chars == 0
    is_very_short = (n_chars < very_short_char_threshold) or (n_tokens < very_short_token_threshold)

    is_partial, is_truncated_like = classify_partial(stripped)

    return {
        "text": raw,
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "has_link": int(n_links > 0),
        "n_links": n_links,
        "is_empty": int(is_empty),
        "is_very_short": int((not is_empty) and is_very_short),
        "is_partial": int((not is_empty) and is_partial),
        "is_truncated_like": int((not is_empty) and is_truncated_like),
    }


def load_corpus_rows(path: Path, cfg: Config) -> pd.DataFrame:
    obj = safe_read_json(path)
    rows: list[dict[str, object]] = []

    for idx, text in iter_candidate_texts(obj):
        audit = audit_text(
            text,
            very_short_char_threshold=cfg.very_short_char_threshold,
            very_short_token_threshold=cfg.very_short_token_threshold,
        )
        rows.append(
            {
                "corpus": path.stem,
                "source_file": str(path),
                "text_index": idx,
                **audit,
            }
        )

    return pd.DataFrame(rows)


def build_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()

    out = (
        rows_df.groupby("corpus", dropna=False)
        .agg(
            n_texts=("text_index", "size"),
            mean_chars=("n_chars", "mean"),
            mean_tokens=("n_tokens", "mean"),
            link_share=("has_link", "mean"),
            mean_links=("n_links", "mean"),
            empty_share=("is_empty", "mean"),
            very_short_share=("is_very_short", "mean"),
            partial_share=("is_partial", "mean"),
            truncated_like_share=("is_truncated_like", "mean"),
        )
        .reset_index()
    )

    for col in [
        "mean_chars",
        "mean_tokens",
        "link_share",
        "mean_links",
        "empty_share",
        "very_short_share",
        "partial_share",
        "truncated_like_share",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.sort_values("corpus").reset_index(drop=True)


def summarize_text(rows_df: pd.DataFrame, summary_df: pd.DataFrame, cfg: Config) -> str:
    lines: list[str] = []
    lines.append("=== Corpus Response Hygiene Audit Summary ===")
    lines.append("")
    lines.append(f"very_short_char_threshold = {cfg.very_short_char_threshold}")
    lines.append(f"very_short_token_threshold = {cfg.very_short_token_threshold}")
    lines.append(f"n_rows = {len(rows_df)}")
    lines.append("")

    if summary_df.empty:
        lines.append("No corpus rows found.")
        return "\n".join(lines)

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['corpus']}: "
            f"n_texts={int(row['n_texts'])}, "
            f"mean_chars={row['mean_chars']:.2f}, "
            f"mean_tokens={row['mean_tokens']:.2f}, "
            f"link_share={row['link_share']:.4f}, "
            f"mean_links={row['mean_links']:.4f}, "
            f"empty_share={row['empty_share']:.4f}, "
            f"very_short_share={row['very_short_share']:.4f}, "
            f"partial_share={row['partial_share']:.4f}, "
            f"truncated_like_share={row['truncated_like_share']:.4f}"
        )

        sub = rows_df[rows_df["corpus"] == row["corpus"]].copy()
        examples = sub[
            (sub["has_link"] == 1)
            | (sub["is_empty"] == 1)
            | (sub["is_very_short"] == 1)
            | (sub["is_partial"] == 1)
            | (sub["is_truncated_like"] == 1)
        ].head(cfg.show_examples_per_corpus)

        if not examples.empty:
            lines.append("  examples:")
            for _, ex in examples.iterrows():
                snippet = str(ex["text"]).strip().replace("\n", " ")
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                flags = []
                for col in ["has_link", "is_empty", "is_very_short", "is_partial", "is_truncated_like"]:
                    if int(ex[col]) == 1:
                        flags.append(col)
                lines.append(
                    f"    - idx={int(ex['text_index'])} | flags={','.join(flags) if flags else 'none'} | text={snippet!r}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit link / empty / partial response hygiene across corpora.")
    parser.add_argument("--corpora-root", default="observatory/corpora")
    parser.add_argument("--outdir", default="outputs/corpus_response_hygiene_audit")
    parser.add_argument("--very-short-char-threshold", type=int, default=40)
    parser.add_argument("--very-short-token-threshold", type=int, default=8)
    parser.add_argument("--show-examples-per-corpus", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(
        corpora_root=args.corpora_root,
        outdir=args.outdir,
        very_short_char_threshold=args.very_short_char_threshold,
        very_short_token_threshold=args.very_short_token_threshold,
        show_examples_per_corpus=args.show_examples_per_corpus,
    )

    corpora_root = Path(cfg.corpora_root)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not corpora_root.exists():
        raise FileNotFoundError(corpora_root)

    json_files = [
        p for p in sorted(corpora_root.glob("*.json"))
        if p.stem in {"C", "Cp", "Cp2", "Cp3", "Cp4"}
    ]
    if not json_files:
        raise FileNotFoundError(f"No corpus variant .json files found in {corpora_root}")

    row_frames: list[pd.DataFrame] = []
    for path in json_files:
        df = load_corpus_rows(path, cfg)
        row_frames.append(df)

    rows_df = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    summary_df = build_summary(rows_df)

    rows_csv = outdir / "corpus_response_hygiene_rows.csv"
    summary_csv = outdir / "corpus_response_hygiene_summary.csv"
    summary_txt = outdir / "corpus_response_hygiene_summary.txt"

    rows_df.to_csv(rows_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    summary_txt.write_text(summarize_text(rows_df, summary_df, cfg), encoding="utf-8")

    print(rows_csv)
    print(summary_csv)
    print(summary_txt)


if __name__ == "__main__":
    main()
