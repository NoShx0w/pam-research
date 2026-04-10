#!/usr/bin/env python3
"""
OBS-028c — Export canonical seam bundle.

Freeze the seam program (OBS-023 through OBS-028b) into one stable bundle for
downstream observatory work.

Exports
-------
<outdir>/
  seam_nodes.csv
  seam_family_summary.csv
  seam_embedding_summary.csv
  seam_metadata.txt

Primary purpose
---------------
Create one canonical seam-side substrate containing:

- relational obstruction field
- anisotropy field
- hotspot classes
- family occupancy summary
- embedding policy summary

Inputs
------
outputs/obs022_scene_bundle/scene_nodes.csv
outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv
outputs/fim_response_operator_decomposition/response_operator_decomposition_nodes.csv
outputs/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_nodes.csv
outputs/obs026_family_two_field_occupancy/family_two_field_class_summary.csv
outputs/obs027_seam_regime_synthesis/obs027_seam_regime_synthesis_summary.txt
outputs/obs028_embedding_comparison/embedding_scores.csv
outputs/obs028b_diffusion_mode_analysis/diffusion_mode_scores.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    scene_nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    mismatch_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    decomposition_csv: str = "outputs/fim_response_operator_decomposition/response_operator_decomposition_nodes.csv"
    obs025_csv: str = "outputs/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_nodes.csv"
    family_summary_csv: str = "outputs/obs026_family_two_field_occupancy/family_two_field_class_summary.csv"
    obs027_summary_txt: str = "outputs/obs027_seam_regime_synthesis/obs027_seam_regime_synthesis_summary.txt"
    embedding_scores_csv: str = "outputs/obs028_embedding_comparison/embedding_scores.csv"
    diffusion_scores_csv: str = "outputs/obs028b_diffusion_mode_analysis/diffusion_mode_scores.csv"
    outdir: str = "outputs/obs028c_canonical_seam_bundle"
    seam_threshold: float = 0.15
    hotspot_quantile: float = 0.85


CANONICAL_NODE_COLS = [
    "node_id",
    "r",
    "alpha",
    "mds1",
    "mds2",
    "signed_phase",
    "distance_to_seam",
    "lazarus_score",
    "response_strength",
    "node_holonomy_proxy",
    "local_direction_mismatch_deg",
    "neighbor_direction_mismatch_mean",
    "transport_align_mean_deg",
    "sym_traceless_norm",
    "scalar_norm",
    "antisymmetric_norm",
    "commutator_norm_rsp",
    "anisotropy_hotspot",
    "relational_hotspot",
    "shared_hotspot",
]

CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    for col in df.columns:
        if col not in {"path_id", "path_family", "route_class", "hotspot_class", "dominant_component"}:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def first_present(df: pd.DataFrame, choices: list[str]) -> str | None:
    for c in choices:
        if c in df.columns:
            return c
    return None


def build_seam_nodes(cfg: Config) -> pd.DataFrame:
    base = load_csv(cfg.scene_nodes_csv)
    mm = load_csv(cfg.mismatch_csv)
    dec = load_csv(cfg.decomposition_csv)
    o25 = load_csv(cfg.obs025_csv)

    out = base.copy()

    # OBS-023 mismatch fields
    mm_cols = [
        c for c in [
            "node_id",
            "local_direction_mismatch_deg",
            "neighbor_direction_mismatch_mean",
            "neighbor_direction_mismatch_deg",
            "transport_align_mean_deg",
        ] if c in mm.columns
    ]
    if "node_id" in mm_cols:
        mm_use = mm[mm_cols].drop_duplicates(subset=["node_id"])
        # normalize naming if needed
        if "neighbor_direction_mismatch_deg" in mm_use.columns and "neighbor_direction_mismatch_mean" not in mm_use.columns:
            mm_use = mm_use.rename(columns={"neighbor_direction_mismatch_deg": "neighbor_direction_mismatch_mean"})
        out = out.merge(mm_use, on="node_id", how="left")

    # response decomposition fields
    dec_cols = [
        c for c in [
            "node_id",
            "sym_traceless_norm",
            "scalar_norm",
            "antisymmetric_norm",
            "commutator_norm_rsp",
        ] if c in dec.columns
    ]
    if "node_id" in dec_cols:
        out = out.merge(dec[dec_cols].drop_duplicates(subset=["node_id"]), on="node_id", how="left")

    # obs025 hotspot fields and any stabilized copies
    o25_cols = [
        c for c in [
            "node_id",
            "sym_traceless_norm",
            "neighbor_direction_mismatch_mean",
            "anisotropy_hotspot",
            "relational_hotspot",
            "shared_hotspot",
        ] if c in o25.columns
    ]
    if "node_id" in o25_cols:
        out = out.merge(
            o25[o25_cols].drop_duplicates(subset=["node_id"]),
            on="node_id",
            how="left",
            suffixes=("", "_obs025"),
        )

        for col in ["sym_traceless_norm", "neighbor_direction_mismatch_mean", "anisotropy_hotspot", "relational_hotspot", "shared_hotspot"]:
            alt = f"{col}_obs025"
            if alt in out.columns:
                if col not in out.columns:
                    out[col] = out[alt]
                else:
                    out[col] = out[col].where(out[col].notna(), out[alt])
                out = out.drop(columns=[alt])

    # derive hotspots if still missing
    if "anisotropy_hotspot" not in out.columns and "sym_traceless_norm" in out.columns:
        thr = float(pd.to_numeric(out["sym_traceless_norm"], errors="coerce").quantile(cfg.hotspot_quantile))
        out["anisotropy_hotspot"] = (pd.to_numeric(out["sym_traceless_norm"], errors="coerce") >= thr).astype(int)

    if "relational_hotspot" not in out.columns:
        rel_col = first_present(out, ["neighbor_direction_mismatch_mean", "transport_align_mean_deg"])
        if rel_col is not None:
            thr = float(pd.to_numeric(out[rel_col], errors="coerce").quantile(cfg.hotspot_quantile))
            out["relational_hotspot"] = (pd.to_numeric(out[rel_col], errors="coerce") >= thr).astype(int)
        else:
            out["relational_hotspot"] = 0

    if "shared_hotspot" not in out.columns:
        out["shared_hotspot"] = (
            (pd.to_numeric(out["anisotropy_hotspot"], errors="coerce").fillna(0) == 1)
            & (pd.to_numeric(out["relational_hotspot"], errors="coerce").fillna(0) == 1)
        ).astype(int)

    keep = [c for c in CANONICAL_NODE_COLS if c in out.columns]
    out = out[keep].copy()

    if "distance_to_seam" in out.columns:
        seam_mask = pd.to_numeric(out["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
        out["seam_band"] = seam_mask.astype(int)

    hotspot_class = np.full(len(out), "non_hotspot", dtype=object)
    aniso = pd.to_numeric(out.get("anisotropy_hotspot", 0), errors="coerce").fillna(0).astype(int)
    rel = pd.to_numeric(out.get("relational_hotspot", 0), errors="coerce").fillna(0).astype(int)
    shared = pd.to_numeric(out.get("shared_hotspot", 0), errors="coerce").fillna(0).astype(int)
    hotspot_class[(aniso == 1) & (shared == 0)] = "anisotropy_only"
    hotspot_class[(rel == 1) & (shared == 0)] = "relational_only"
    hotspot_class[shared == 1] = "shared"
    out["hotspot_class"] = hotspot_class

    return out.sort_values(["r", "alpha"]).reset_index(drop=True)


def build_family_summary(cfg: Config) -> pd.DataFrame:
    fam = load_csv(cfg.family_summary_csv)
    if "route_class" in fam.columns:
        order = {k: i for i, k in enumerate(CLASS_ORDER)}
        fam["order"] = fam["route_class"].map(lambda x: order.get(x, 999))
        fam = fam.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return fam


def build_embedding_summary(cfg: Config) -> pd.DataFrame:
    emb = load_csv(cfg.embedding_scores_csv)
    diff = load_csv(cfg.diffusion_scores_csv)

    rows = []

    if len(emb):
        best_seam = emb.sort_values("seam_separation", ascending=False).iloc[0]
        best_family = emb.sort_values("family_centroid_separation", ascending=False).iloc[0]
        best_trust = emb.sort_values("trustworthiness", ascending=False).iloc[0]
        rows.extend(
            [
                {"source": "obs028", "metric": "best_seam_separation", "value": best_seam["seam_separation"], "label": str(best_seam["embedding"])},
                {"source": "obs028", "metric": "best_family_separation", "value": best_family["family_centroid_separation"], "label": str(best_family["embedding"])},
                {"source": "obs028", "metric": "best_trustworthiness", "value": best_trust["trustworthiness"], "label": str(best_trust["embedding"])},
            ]
        )

    if len(diff):
        best_seam_t = diff.sort_values("seam_separation", ascending=False).iloc[0]
        best_branch_t = diff.sort_values("axis_align_branch_exit", ascending=False).iloc[0]
        rows.extend(
            [
                {"source": "obs028b", "metric": "best_diffusion_seam_time", "value": best_seam_t["seam_separation"], "label": f"t={int(best_seam_t['diffusion_time'])}"},
                {"source": "obs028b", "metric": "best_diffusion_branch_time", "value": best_branch_t["axis_align_branch_exit"], "label": f"t={int(best_branch_t['diffusion_time'])}"},
            ]
        )

    return pd.DataFrame(rows)


def write_metadata(cfg: Config, seam_nodes: pd.DataFrame, fam: pd.DataFrame, emb: pd.DataFrame) -> str:
    seam_mask = pd.to_numeric(seam_nodes.get("distance_to_seam"), errors="coerce") <= cfg.seam_threshold
    aniso_only = int(((seam_nodes["hotspot_class"] == "anisotropy_only")).sum()) if "hotspot_class" in seam_nodes.columns else 0
    rel_only = int(((seam_nodes["hotspot_class"] == "relational_only")).sum()) if "hotspot_class" in seam_nodes.columns else 0
    shared = int(((seam_nodes["hotspot_class"] == "shared")).sum()) if "hotspot_class" in seam_nodes.columns else 0

    lines = [
        "=== OBS-028c Canonical Seam Bundle Metadata ===",
        "",
        "Bundle purpose",
        "- freeze the seam arc (OBS-023 through OBS-028b) into one reusable observatory substrate",
        "",
        "Counts",
        f"n_nodes={len(seam_nodes)}",
        f"n_seam_band_nodes={int(seam_mask.sum()) if seam_mask.notna().any() else 0}",
        f"n_anisotropy_only_hotspots={aniso_only}",
        f"n_relational_only_hotspots={rel_only}",
        f"n_shared_hotspots={shared}",
        f"n_family_rows={len(fam)}",
        f"n_embedding_summary_rows={len(emb)}",
        "",
        "Canonical seam interpretation",
        "- the seam is a multi-field structural regime",
        "- relational obstruction is the stronger seam discriminator",
        "- response anisotropy is a distinct seam-side field",
        "- families differ by residency pattern within this regime",
        "- diffusion exposes seam distance as a dominant slow mode",
        "",
        "Recommended embedding policy",
        "- MDS remains canonical for geometry-faithful observatory layout",
        "- UMAP is the strongest supplementary exploratory embedding",
        "- diffusion is best treated as a slow-mode diagnostic rather than primary display coordinates",
        "",
        "Primary node fields",
    ]

    for c in seam_nodes.columns:
        lines.append(f"  - {c}")

    lines.extend(["", "Inputs"])
    for label, path in [
        ("scene_nodes_csv", cfg.scene_nodes_csv),
        ("mismatch_csv", cfg.mismatch_csv),
        ("decomposition_csv", cfg.decomposition_csv),
        ("obs025_csv", cfg.obs025_csv),
        ("family_summary_csv", cfg.family_summary_csv),
        ("obs027_summary_txt", cfg.obs027_summary_txt),
        ("embedding_scores_csv", cfg.embedding_scores_csv),
        ("diffusion_scores_csv", cfg.diffusion_scores_csv),
    ]:
        lines.append(f"{label}={path}")

    # include obs027 summary text if present
    p = Path(cfg.obs027_summary_txt)
    if p.exists():
        lines.extend(["", "OBS-027 summary excerpt"])
        try:
            text = p.read_text(encoding="utf-8").strip().splitlines()
            lines.extend(text[:20])
        except Exception:
            pass

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical seam bundle.")
    parser.add_argument("--scene-nodes-csv", default=Config.scene_nodes_csv)
    parser.add_argument("--mismatch-csv", default=Config.mismatch_csv)
    parser.add_argument("--decomposition-csv", default=Config.decomposition_csv)
    parser.add_argument("--obs025-csv", default=Config.obs025_csv)
    parser.add_argument("--family-summary-csv", default=Config.family_summary_csv)
    parser.add_argument("--obs027-summary-txt", default=Config.obs027_summary_txt)
    parser.add_argument("--embedding-scores-csv", default=Config.embedding_scores_csv)
    parser.add_argument("--diffusion-scores-csv", default=Config.diffusion_scores_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    args = parser.parse_args()

    cfg = Config(
        scene_nodes_csv=args.scene_nodes_csv,
        mismatch_csv=args.mismatch_csv,
        decomposition_csv=args.decomposition_csv,
        obs025_csv=args.obs025_csv,
        family_summary_csv=args.family_summary_csv,
        obs027_summary_txt=args.obs027_summary_txt,
        embedding_scores_csv=args.embedding_scores_csv,
        diffusion_scores_csv=args.diffusion_scores_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        hotspot_quantile=args.hotspot_quantile,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seam_nodes = build_seam_nodes(cfg)
    fam = build_family_summary(cfg)
    emb = build_embedding_summary(cfg)
    metadata = write_metadata(cfg, seam_nodes, fam, emb)

    seam_nodes.to_csv(outdir / "seam_nodes.csv", index=False)
    fam.to_csv(outdir / "seam_family_summary.csv", index=False)
    emb.to_csv(outdir / "seam_embedding_summary.csv", index=False)
    (outdir / "seam_metadata.txt").write_text(metadata, encoding="utf-8")

    print(outdir / "seam_nodes.csv")
    print(outdir / "seam_family_summary.csv")
    print(outdir / "seam_embedding_summary.csv")
    print(outdir / "seam_metadata.txt")


if __name__ == "__main__":
    main()
