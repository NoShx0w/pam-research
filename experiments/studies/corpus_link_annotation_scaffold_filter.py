from __future__ import annotations

"""
Filter the corpus link annotation scaffold down to link-bearing rows only.

Purpose
-------
Create a smaller CSV for manual annotation of the link phenomenon
(geometric adjacency / pressure-valve hypothesis) by keeping only rows
where `has_link == 1`.

Expected input
--------------
outputs/corpus_link_annotation/corpus_link_annotation_scaffold.csv

Outputs
-------
<outdir>/
  corpus_link_annotation_links_only.csv
  corpus_link_annotation_links_only_summary.txt
"""

import argparse
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Corpus Link Annotation (Links Only) Summary ===")
    lines.append("")
    lines.append(f"n_rows = {len(df)}")
    lines.append("")

    if df.empty:
        lines.append("No link-bearing rows found.")
        return "\n".join(lines) + "\n"

    for corpus, sub in df.groupby("corpus", dropna=False):
        lines.append(
            f"{corpus}: "
            f"n_rows={len(sub)}, "
            f"mean_links={pd.to_numeric(sub['n_links'], errors='coerce').mean():.4f}, "
            f"boundary_none={(sub['boundary_event'] == 'none').mean():.4f}, "
            f"boundary_empty={(sub['boundary_event'] == 'empty_session_failure').mean():.4f}, "
            f"boundary_partial={(sub['boundary_event'] == 'partial_session_failure').mean():.4f}"
        )

    lines.append("")
    lines.append("Suggested annotation columns:")
    lines.append("- link_adj_label")
    lines.append("- link_adj_strength")
    lines.append("- anchor_type")
    lines.append("- thumb_relevant")
    lines.append("- pressure_valve_judgment")
    lines.append("- notes")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter annotation scaffold to link-bearing rows only.")
    parser.add_argument(
        "--input-csv",
        default="outputs/corpus_link_annotation/corpus_link_annotation_scaffold.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/corpus_link_annotation",
    )
    parser.add_argument(
        "--sort-by",
        nargs="*",
        default=["corpus", "text_index"],
        help="Columns to sort by before writing output.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    if "has_link" not in df.columns:
        raise ValueError(f"`has_link` column not found in {input_csv}")

    links_only = df[pd.to_numeric(df["has_link"], errors="coerce").fillna(0).astype(int) == 1].copy()

    sort_cols = [c for c in args.sort_by if c in links_only.columns]
    if sort_cols:
        links_only = links_only.sort_values(sort_cols).reset_index(drop=True)

    out_csv = outdir / "corpus_link_annotation_links_only.csv"
    out_txt = outdir / "corpus_link_annotation_links_only_summary.txt"

    links_only.to_csv(out_csv, index=False)
    out_txt.write_text(summarize(links_only), encoding="utf-8")

    print(out_csv)
    print(out_txt)


if __name__ == "__main__":
    main()
