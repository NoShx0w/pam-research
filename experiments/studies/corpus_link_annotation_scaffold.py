from __future__ import annotations


"""
Build a first-pass annotation scaffold for link / boundary-event analysis
across PAM corpus variants.

Purpose
-------
Create a response-level CSV suitable for manual annotation of:
- link presence
- geometric / topical adjacency
- pressure-valve judgments
- known boundary events

This script does NOT decide the scientific interpretation.
It only prepares a clean annotation table with auto-filled structural fields.

Expected input
--------------
A directory containing corpus JSON files such as:

    observatory/corpora/
      C.json
      Cp.json
      Cp2.json
      Cp3.json
      Cp4.json

Each corpus file is expected to be a JSON list of strings.

Outputs
-------
<outdir>/
  corpus_link_annotation_scaffold.csv
  corpus_link_annotation_scaffold_summary.txt

Notes
-----
- boundary_event defaults to "none"
- manual annotation columns are left blank where appropriate
- known boundary events can be injected via CLI flags if desired
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

DEFAULT_CORPORA = ("C", "Cp", "Cp2", "Cp3", "Cp4")


@dataclass(frozen=True)
class Config:
    corpora_root: str = "observatory/corpora"
    outdir: str = "outputs/corpus_link_annotation"
    annotator: str = ""
    corpora: tuple[str, ...] = DEFAULT_CORPORA
    cp3_empty_index: int | None = 49
    cp4_failure_start_index: int | None = 38


def safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_texts(obj: Any) -> list[str]:
    if isinstance(obj, list) and all(isinstance(x, str) or x is None for x in obj):
        return ["" if x is None else str(x) for x in obj]
    raise ValueError(f"Unsupported corpus JSON structure for annotation scaffold: {type(obj)}")


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def extract_links(text: str) -> list[str]:
    matches = URL_RE.findall(text or "")
    return [m.strip() for m in matches if str(m).strip()]


def extract_domains(links: list[str]) -> list[str]:
    domains: list[str] = []
    for link in links:
        normalized = link
        if normalized.startswith("www."):
            normalized = "http://" + normalized
        m = re.match(r"(?i)https?://([^/\s]+)", normalized)
        if not m:
            continue
        domain = m.group(1).lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        domains.append(domain)
    # stable unique order
    seen: set[str] = set()
    out: list[str] = []
    for d in domains:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def classify_length_band(n_tokens: int) -> str:
    if n_tokens < 120:
        return "short"
    if n_tokens < 300:
        return "medium"
    return "long"


def boundary_event_default(corpus: str, idx: int, text: str, cfg: Config) -> str:
    """
    Provenance-aware defaults based on known session failures discussed by user.

    Current known cases:
    - Cp3 had a bricked/frozen chat after 49 responses; empty final response residue
    - Cp4 had a bricked/frozen/unusable chat after 38 responses; later responses may be partial/empty residues

    These are defaults only and can be revised manually.
    """
    stripped = (text or "").strip()

    if corpus == "Cp3" and cfg.cp3_empty_index is not None and idx == cfg.cp3_empty_index and stripped == "":
        return "empty_session_failure"

    if corpus == "Cp4" and cfg.cp4_failure_start_index is not None and idx >= cfg.cp4_failure_start_index:
        if stripped == "":
            return "empty_session_failure"
        return "partial_session_failure"

    return "none"


def build_rows(corpus_name: str, texts: list[str], cfg: Config) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for idx, raw_text in enumerate(texts):
        text = "" if raw_text is None else str(raw_text)
        stripped = text.strip()

        links = extract_links(text)
        domains = extract_domains(links)

        n_chars = len(stripped)
        n_tokens = token_count(stripped)
        has_link = int(len(links) > 0)
        n_links = len(links)
        response_id = f"{corpus_name}:{idx}"

        row = {
            # identity
            "corpus": corpus_name,
            "text_index": idx,
            "response_id": response_id,

            # raw structure
            "response_text": text,
            "n_chars": n_chars,
            "n_tokens": n_tokens,
            "has_link": has_link,
            "n_links": n_links,
            "domains": ";".join(domains),
            "turn_depth_proxy": idx + 1,
            "length_band": classify_length_band(n_tokens),

            # provenance / boundary
            "boundary_event": boundary_event_default(corpus_name, idx, text, cfg),

            # manual annotation columns
            "link_adj_label": "",
            "link_adj_strength": "",
            "anchor_type": "",
            "thumb_relevant": "",
            "pressure_valve_judgment": "",
            "annotator": cfg.annotator,
            "notes": "",
        }
        rows.append(row)

    return rows


def summarize(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Corpus Link Annotation Scaffold Summary ===")
    lines.append("")
    lines.append(f"n_rows = {len(df)}")
    lines.append("")

    if df.empty:
        lines.append("No rows generated.")
        return "\n".join(lines)

    for corpus, sub in df.groupby("corpus", dropna=False):
        n = len(sub)
        lines.append(
            f"{corpus}: "
            f"n_texts={n}, "
            f"link_share={sub['has_link'].mean():.4f}, "
            f"mean_links={sub['n_links'].mean():.4f}, "
            f"boundary_none={(sub['boundary_event'] == 'none').mean():.4f}, "
            f"boundary_empty={(sub['boundary_event'] == 'empty_session_failure').mean():.4f}, "
            f"boundary_partial={(sub['boundary_event'] == 'partial_session_failure').mean():.4f}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build annotation scaffold for corpus link / boundary-event study.")
    parser.add_argument("--corpora-root", default="observatory/corpora")
    parser.add_argument("--outdir", default="outputs/corpus_link_annotation")
    parser.add_argument("--annotator", default="")
    parser.add_argument("--cp3-empty-index", type=int, default=49)
    parser.add_argument("--cp4-failure-start-index", type=int, default=38)
    parser.add_argument(
        "--corpora",
        nargs="*",
        default=list(DEFAULT_CORPORA),
        help="Corpus stems to include, e.g. C Cp Cp2 Cp3 Cp4",
    )
    args = parser.parse_args()

    cfg = Config(
        corpora_root=args.corpora_root,
        outdir=args.outdir,
        annotator=args.annotator,
        corpora=tuple(args.corpora),
        cp3_empty_index=args.cp3_empty_index,
        cp4_failure_start_index=args.cp4_failure_start_index,
    )

    corpora_root = Path(cfg.corpora_root)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not corpora_root.exists():
        raise FileNotFoundError(corpora_root)

    rows: list[dict[str, object]] = []
    for corpus_name in cfg.corpora:
        path = corpora_root / f"{corpus_name}.json"
        if not path.exists():
            raise FileNotFoundError(path)

        obj = safe_read_json(path)
        texts = extract_texts(obj)
        rows.extend(build_rows(corpus_name, texts, cfg))

    df = pd.DataFrame(rows)

    desired_cols = [
        "corpus",
        "text_index",
        "response_id",
        "response_text",
        "n_chars",
        "n_tokens",
        "has_link",
        "n_links",
        "domains",
        "turn_depth_proxy",
        "length_band",
        "boundary_event",
        "link_adj_label",
        "link_adj_strength",
        "anchor_type",
        "thumb_relevant",
        "pressure_valve_judgment",
        "annotator",
        "notes",
    ]
    df = df[desired_cols].copy()

    out_csv = outdir / "corpus_link_annotation_scaffold.csv"
    out_txt = outdir / "corpus_link_annotation_scaffold_summary.txt"

    df.to_csv(out_csv, index=False)
    out_txt.write_text(summarize(df), encoding="utf-8")

    print(out_csv)
    print(out_txt)


if __name__ == "__main__":
    main()
