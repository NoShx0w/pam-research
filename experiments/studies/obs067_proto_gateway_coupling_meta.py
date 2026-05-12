#!/usr/bin/env python3
"""
OBS-067 — Proto-to-gateway/canonical survival coupling meta-analysis.

This study consumes compact OBS-065 and OBS-066 summary outputs for C and Cp.
It does not rescan path-level symbolic traces. It asks whether downstream
gateway/canonical-family survival tracks, absorbs, or amplifies proto-groupoid
survival drift under the same route-origin decoy replacement regimes.

Expected inputs, per corpus label X:

  outputs/obs065_proto_groupoid_decoy_survival_X/
    obs065_proto_survival_summary.csv
    obs065_cross_layer_failure_modes.csv
    obs065_proto_anchor_candidates.csv

  outputs/obs066_gateway_canonical_family_survival_X/
    obs066_gateway_survival_summary.csv
    obs066_canonical_family_survival_summary.csv
    obs066_anchor_family_survival_summary.csv
    obs066_cross_layer_consequence_modes.csv

Default output:

  outputs/obs067_proto_gateway_coupling_meta/
    obs067_layer_coupling_table.csv
    obs067_projection_absorption_modes.csv
    obs067_proto_sufficiency_table.csv
    obs067_anchor_transfer_table.csv
    obs067_cross_corpus_contrast.csv
    obs067_proto_gateway_coupling_meta_report.md
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STUDY_ID = "OBS-067"
STUDY_TITLE = "Proto-to-gateway/canonical survival coupling meta-analysis"
DEFAULT_CORPORA = ("C", "Cp")
KEY_COLS = ["corpus_label", "band", "replacement_route_class"]

PROTO_LAYERS = [
    "generator_completed",
    "composition",
    "proto_edge",
    "proto_sector_edge",
    "proto_relation",
    "proto_sector_relation",
]

GATEWAY_LAYERS = [
    "gateway_event",
    "gateway_event_generator",
    "gateway_event_relation",
    "gateway_event_generator_relation",
]

CANONICAL_LAYERS = [
    "canonical_family",
    "canonical_family_relation",
]

ANCHOR_LAYERS = [
    "anchor_relation_indicator",
    "anchor_canonical_family",
]

ALL_SURVIVAL_LAYERS = PROTO_LAYERS + GATEWAY_LAYERS + CANONICAL_LAYERS + ANCHOR_LAYERS


@dataclass(frozen=True)
class InputPaths:
    corpus_label: str
    obs065_dir: Path
    obs066_dir: Path

    @property
    def proto_summary(self) -> Path:
        return self.obs065_dir / "obs065_proto_survival_summary.csv"

    @property
    def proto_failure_modes(self) -> Path:
        return self.obs065_dir / "obs065_cross_layer_failure_modes.csv"

    @property
    def proto_anchor_candidates(self) -> Path:
        return self.obs065_dir / "obs065_proto_anchor_candidates.csv"

    @property
    def gateway_summary(self) -> Path:
        return self.obs066_dir / "obs066_gateway_survival_summary.csv"

    @property
    def canonical_summary(self) -> Path:
        return self.obs066_dir / "obs066_canonical_family_survival_summary.csv"

    @property
    def anchor_summary(self) -> Path:
        return self.obs066_dir / "obs066_anchor_family_survival_summary.csv"

    @property
    def consequence_modes(self) -> Path:
        return self.obs066_dir / "obs066_cross_layer_consequence_modes.csv"


@dataclass(frozen=True)
class StudyConfig:
    root: Path
    out_dir: Path
    corpora: tuple[str, ...]
    survival_high: float
    survival_low: float
    delta_threshold: float
    anchor_threshold: float
    top_anchor_n: int


def parse_args() -> StudyConfig:
    parser = argparse.ArgumentParser(
        description="OBS-067 proto-to-gateway/canonical coupling meta-analysis."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/obs067_proto_gateway_coupling_meta"),
        help="Output directory, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--corpora",
        default=",".join(DEFAULT_CORPORA),
        help="Comma-separated corpus labels. Default: C,Cp.",
    )
    parser.add_argument(
        "--survival-high",
        type=float,
        default=0.75,
        help="Threshold used to classify high survival. Default: 0.75.",
    )
    parser.add_argument(
        "--survival-low",
        type=float,
        default=0.25,
        help="Threshold used to classify low survival. Default: 0.25.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.25,
        help="Minimum survival-rate delta for absorb/amplify classifications. Default: 0.25.",
    )
    parser.add_argument(
        "--anchor-threshold",
        type=float,
        default=0.75,
        help="Minimum anchor score/survival used for strong anchor summaries. Default: 0.75.",
    )
    parser.add_argument(
        "--top-anchor-n",
        type=int,
        default=20,
        help="Number of strongest anchor candidates to preserve per corpus. Default: 20.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    corpora = tuple(c.strip() for c in args.corpora.split(",") if c.strip())
    if not corpora:
        raise SystemExit("No corpus labels supplied.")

    return StudyConfig(
        root=root,
        out_dir=out_dir,
        corpora=corpora,
        survival_high=args.survival_high,
        survival_low=args.survival_low,
        delta_threshold=args.delta_threshold,
        anchor_threshold=args.anchor_threshold,
        top_anchor_n=args.top_anchor_n,
    )


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required input is empty: {path}")


def input_paths_for(config: StudyConfig, corpus_label: str) -> InputPaths:
    return InputPaths(
        corpus_label=corpus_label,
        obs065_dir=config.root / f"outputs/obs065_proto_groupoid_decoy_survival_{corpus_label}",
        obs066_dir=config.root / f"outputs/obs066_gateway_canonical_family_survival_{corpus_label}",
    )


def validate_inputs(paths: InputPaths) -> None:
    for path in [
        paths.proto_summary,
        paths.proto_failure_modes,
        paths.proto_anchor_candidates,
        paths.gateway_summary,
        paths.canonical_summary,
        paths.anchor_summary,
        paths.consequence_modes,
    ]:
        require_file(path)


def read_summary(path: Path, corpus_label: str, allowed_layers: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {
        "band",
        "replacement_route_class",
        "layer",
        "top1_survival_rate",
        "mean_top3_overlap_share",
        "median_distribution_tv_distance",
        "p90_distribution_tv_distance",
        "baseline_top1_label",
        "baseline_top1_minus_top2_margin",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df[df["layer"].isin(list(allowed_layers))].copy()
    df.insert(0, "corpus_label", corpus_label)
    return df


def pivot_metric(df: pd.DataFrame, metric: str, prefix: str = "") -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Cannot pivot missing metric: {metric}")
    wide = (
        df.pivot_table(
            index=KEY_COLS,
            columns="layer",
            values=metric,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename = {layer: f"{prefix}{layer}_{metric}" for layer in ALL_SURVIVAL_LAYERS if layer in wide.columns}
    return wide.rename(columns=rename)


def pivot_label(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = (
        df.pivot_table(
            index=KEY_COLS,
            columns="layer",
            values=metric,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename = {layer: f"{layer}_{metric}" for layer in ALL_SURVIVAL_LAYERS if layer in wide.columns}
    return wide.rename(columns=rename)


def safe_num(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def classify_projection_absorption(row: pd.Series, config: StudyConfig) -> str:
    pr = safe_num(row.get("proto_relation_top1_survival_rate"))
    psr = safe_num(row.get("proto_sector_relation_top1_survival_rate"))
    ge = safe_num(row.get("gateway_event_top1_survival_rate"))
    gr = safe_num(row.get("gateway_event_relation_top1_survival_rate"))
    cr = safe_num(row.get("canonical_family_relation_top1_survival_rate"))
    af = safe_num(row.get("anchor_canonical_family_top1_survival_rate"))

    downstream_mean = np.nanmean([gr, cr])
    if np.isnan(pr) or np.isnan(downstream_mean):
        return "insufficient_data"

    high = config.survival_high
    low = config.survival_low
    delta = config.delta_threshold

    if pr >= high and downstream_mean >= high:
        return "broad_coupled_survival"
    if pr <= low and downstream_mean <= low:
        if not np.isnan(af) and af >= high:
            return "anchor_family_rescues_relation_failure"
        return "broad_coupled_failure"
    if downstream_mean - pr >= delta:
        if not np.isnan(psr) and psr >= high and pr <= low:
            return "sector_projection_absorbs_fine_proto_drift"
        return "downstream_absorbs_proto_drift"
    if pr - downstream_mean >= delta:
        return "proto_survives_downstream_fails"
    if not np.isnan(ge) and ge >= high and pr <= low and downstream_mean <= low:
        return "gateway_event_survives_relation_fails"
    if not np.isnan(af) and af - pr >= delta:
        return "anchor_family_rescues_relation_failure"
    return "mixed_or_ambiguous"


def classify_proto_sufficiency(row: pd.Series, config: StudyConfig) -> str:
    generator = safe_num(row.get("generator_completed_top1_survival_rate"))
    proto_edge = safe_num(row.get("proto_edge_top1_survival_rate"))
    proto_relation = safe_num(row.get("proto_relation_top1_survival_rate"))
    gateway_event = safe_num(row.get("gateway_event_top1_survival_rate"))
    gateway_relation = safe_num(row.get("gateway_event_relation_top1_survival_rate"))
    canonical_relation = safe_num(row.get("canonical_family_relation_top1_survival_rate"))

    high = config.survival_high
    low = config.survival_low

    if generator >= high and proto_edge <= low and proto_relation <= low:
        return "generator_not_sufficient_for_proto_survival"
    if proto_edge >= high and proto_relation <= low:
        return "typed_edge_not_sufficient_for_relation_survival"
    if proto_relation >= high and gateway_relation <= low and canonical_relation <= low:
        return "proto_relation_not_sufficient_for_downstream_survival"
    if proto_relation <= low and gateway_event >= high:
        return "gateway_event_not_sufficient_for_relation_survival"
    if proto_relation >= high and gateway_relation >= high and canonical_relation >= high:
        return "proto_relation_sufficient_in_this_regime"
    return "no_simple_sufficiency_pattern"


def classify_anchor_transfer(row: pd.Series, config: StudyConfig) -> str:
    proto_relation = safe_num(row.get("proto_relation_top1_survival_rate"))
    proto_sector_relation = safe_num(row.get("proto_sector_relation_top1_survival_rate"))
    anchor_indicator = safe_num(row.get("anchor_relation_indicator_top1_survival_rate"))
    anchor_family = safe_num(row.get("anchor_canonical_family_top1_survival_rate"))
    canonical_relation = safe_num(row.get("canonical_family_relation_top1_survival_rate"))

    high = config.survival_high
    low = config.survival_low
    delta = config.delta_threshold

    if np.isnan(anchor_family):
        return "insufficient_anchor_data"
    if anchor_family >= high and proto_relation <= low:
        return "anchor_family_transfers_despite_fine_proto_failure"
    if anchor_family >= high and canonical_relation >= high:
        return "anchor_family_tracks_canonical_survival"
    if anchor_indicator >= high and anchor_family <= low:
        return "anchor_indicator_survives_family_specific_anchor_fails"
    if proto_sector_relation - proto_relation >= delta and anchor_family >= high:
        return "sector_anchor_transfer"
    if anchor_family <= low and proto_relation <= low:
        return "anchor_and_proto_joint_failure"
    return "mixed_anchor_transfer"


def build_layer_coupling_table(config: StudyConfig) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, dict[str, int]] = {}

    for corpus_label in config.corpora:
        paths = input_paths_for(config, corpus_label)
        validate_inputs(paths)

        proto = read_summary(paths.proto_summary, corpus_label, PROTO_LAYERS)
        gateway = read_summary(paths.gateway_summary, corpus_label, GATEWAY_LAYERS)
        canonical = read_summary(paths.canonical_summary, corpus_label, CANONICAL_LAYERS)
        anchor = read_summary(paths.anchor_summary, corpus_label, ANCHOR_LAYERS)
        combined = pd.concat([proto, gateway, canonical, anchor], ignore_index=True)

        survival = pivot_metric(combined, "top1_survival_rate")
        top3 = pivot_metric(combined, "mean_top3_overlap_share")
        tv = pivot_metric(combined, "median_distribution_tv_distance")
        margin = pivot_metric(combined, "baseline_top1_minus_top2_margin")
        labels = pivot_label(combined, "baseline_top1_label")

        wide = survival
        for addon in [top3, tv, margin, labels]:
            wide = wide.merge(addon, on=KEY_COLS, how="outer")

        failure = pd.read_csv(paths.proto_failure_modes)
        failure.insert(0, "corpus_label", corpus_label)
        failure_cols = KEY_COLS + ["failure_mode"]
        wide = wide.merge(failure[failure_cols], on=KEY_COLS, how="left")

        consequence = pd.read_csv(paths.consequence_modes)
        consequence.insert(0, "corpus_label", corpus_label)
        consequence_cols = KEY_COLS + ["consequence_mode"]
        wide = wide.merge(consequence[consequence_cols], on=KEY_COLS, how="left")

        diagnostics[corpus_label] = {
            "proto_summary_rows": int(len(proto)),
            "gateway_summary_rows": int(len(gateway)),
            "canonical_summary_rows": int(len(canonical)),
            "anchor_summary_rows": int(len(anchor)),
            "joined_regime_rows": int(len(wide)),
        }
        frames.append(wide)

    table = pd.concat(frames, ignore_index=True)
    table = add_coupling_metrics(table, config)
    return table, diagnostics


def add_coupling_metrics(table: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    out = table.copy()

    def delta(a: str, b: str) -> pd.Series:
        return out[a] - out[b]

    required = [
        "proto_relation_top1_survival_rate",
        "proto_sector_relation_top1_survival_rate",
        "proto_edge_top1_survival_rate",
        "gateway_event_top1_survival_rate",
        "gateway_event_generator_top1_survival_rate",
        "gateway_event_relation_top1_survival_rate",
        "gateway_event_generator_relation_top1_survival_rate",
        "canonical_family_top1_survival_rate",
        "canonical_family_relation_top1_survival_rate",
        "anchor_relation_indicator_top1_survival_rate",
        "anchor_canonical_family_top1_survival_rate",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected wide survival columns: {missing}")

    out["gateway_relation_minus_proto_relation_delta"] = delta(
        "gateway_event_relation_top1_survival_rate",
        "proto_relation_top1_survival_rate",
    )
    out["canonical_relation_minus_proto_relation_delta"] = delta(
        "canonical_family_relation_top1_survival_rate",
        "proto_relation_top1_survival_rate",
    )
    out["anchor_family_minus_proto_relation_delta"] = delta(
        "anchor_canonical_family_top1_survival_rate",
        "proto_relation_top1_survival_rate",
    )
    out["gateway_event_minus_proto_edge_delta"] = delta(
        "gateway_event_top1_survival_rate",
        "proto_edge_top1_survival_rate",
    )
    out["gateway_event_generator_minus_proto_sector_edge_delta"] = delta(
        "gateway_event_generator_top1_survival_rate",
        "proto_sector_edge_top1_survival_rate",
    )
    out["canonical_relation_minus_proto_sector_relation_delta"] = delta(
        "canonical_family_relation_top1_survival_rate",
        "proto_sector_relation_top1_survival_rate",
    )
    out["sector_relation_minus_proto_relation_delta"] = delta(
        "proto_sector_relation_top1_survival_rate",
        "proto_relation_top1_survival_rate",
    )
    out["gateway_generator_relation_minus_proto_relation_delta"] = delta(
        "gateway_event_generator_relation_top1_survival_rate",
        "proto_relation_top1_survival_rate",
    )

    out["downstream_relation_mean_top1_survival"] = out[
        [
            "gateway_event_relation_top1_survival_rate",
            "canonical_family_relation_top1_survival_rate",
        ]
    ].mean(axis=1)
    out["downstream_projection_mean_top1_survival"] = out[
        [
            "gateway_event_top1_survival_rate",
            "gateway_event_generator_top1_survival_rate",
            "canonical_family_top1_survival_rate",
        ]
    ].mean(axis=1)
    out["fine_proto_mean_top1_survival"] = out[
        ["proto_edge_top1_survival_rate", "proto_relation_top1_survival_rate"]
    ].mean(axis=1)
    out["sector_proto_mean_top1_survival"] = out[
        ["proto_sector_edge_top1_survival_rate", "proto_sector_relation_top1_survival_rate"]
    ].mean(axis=1)

    out["projection_absorption_mode"] = out.apply(
        lambda row: classify_projection_absorption(row, config), axis=1
    )
    out["proto_sufficiency_mode"] = out.apply(
        lambda row: classify_proto_sufficiency(row, config), axis=1
    )
    out["anchor_transfer_mode"] = out.apply(
        lambda row: classify_anchor_transfer(row, config), axis=1
    )

    ordered_cols = KEY_COLS + [
        "projection_absorption_mode",
        "proto_sufficiency_mode",
        "anchor_transfer_mode",
        "failure_mode",
        "consequence_mode",
    ]
    remaining = [c for c in out.columns if c not in ordered_cols]
    return out[ordered_cols + remaining].sort_values(KEY_COLS).reset_index(drop=True)


def build_projection_absorption_modes(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in table.groupby(["corpus_label", "projection_absorption_mode"], dropna=False):
        corpus_label, mode = keys
        rows.append(
            {
                "corpus_label": corpus_label,
                "projection_absorption_mode": mode,
                "n_regimes": int(len(group)),
                "mean_proto_relation_survival": group["proto_relation_top1_survival_rate"].mean(),
                "mean_gateway_relation_survival": group["gateway_event_relation_top1_survival_rate"].mean(),
                "mean_canonical_relation_survival": group["canonical_family_relation_top1_survival_rate"].mean(),
                "mean_anchor_family_survival": group["anchor_canonical_family_top1_survival_rate"].mean(),
                "mean_gateway_relation_minus_proto_delta": group[
                    "gateway_relation_minus_proto_relation_delta"
                ].mean(),
                "mean_canonical_relation_minus_proto_delta": group[
                    "canonical_relation_minus_proto_relation_delta"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["corpus_label", "projection_absorption_mode"])


def build_proto_sufficiency_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in table.groupby(["corpus_label", "proto_sufficiency_mode"], dropna=False):
        corpus_label, mode = keys
        rows.append(
            {
                "corpus_label": corpus_label,
                "proto_sufficiency_mode": mode,
                "n_regimes": int(len(group)),
                "mean_generator_survival": group["generator_completed_top1_survival_rate"].mean(),
                "mean_proto_edge_survival": group["proto_edge_top1_survival_rate"].mean(),
                "mean_proto_relation_survival": group["proto_relation_top1_survival_rate"].mean(),
                "mean_gateway_event_survival": group["gateway_event_top1_survival_rate"].mean(),
                "mean_gateway_relation_survival": group["gateway_event_relation_top1_survival_rate"].mean(),
                "mean_canonical_relation_survival": group["canonical_family_relation_top1_survival_rate"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["corpus_label", "proto_sufficiency_mode"])


def build_anchor_transfer_table(config: StudyConfig, coupling_table: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for corpus_label in config.corpora:
        paths = input_paths_for(config, corpus_label)
        anchors = pd.read_csv(paths.proto_anchor_candidates)
        anchors.insert(0, "corpus_label", corpus_label)

        required = {
            "band",
            "replacement_route_class",
            "evaluated_route_class",
            "baseline_proto_relation",
            "baseline_proto_sector_relation",
            "relation_survival_rate",
            "sector_relation_survival_rate",
            "component_edge_rank_survival_rate",
            "component_generator_rank_survival_rate",
            "anchor_score",
            "anchor_class",
        }
        missing = sorted(required - set(anchors.columns))
        if missing:
            raise ValueError(f"{paths.proto_anchor_candidates} missing columns: {missing}")

        anchors = anchors.merge(
            coupling_table[
                KEY_COLS
                + [
                    "projection_absorption_mode",
                    "proto_sufficiency_mode",
                    "anchor_transfer_mode",
                    "proto_relation_top1_survival_rate",
                    "proto_sector_relation_top1_survival_rate",
                    "gateway_event_relation_top1_survival_rate",
                    "canonical_family_relation_top1_survival_rate",
                    "anchor_canonical_family_top1_survival_rate",
                ]
            ],
            on=KEY_COLS,
            how="left",
        )
        anchors["anchor_relation_to_gateway_relation_delta"] = (
            anchors["gateway_event_relation_top1_survival_rate"]
            - anchors["relation_survival_rate"]
        )
        anchors["anchor_relation_to_canonical_relation_delta"] = (
            anchors["canonical_family_relation_top1_survival_rate"]
            - anchors["relation_survival_rate"]
        )
        anchors["is_strong_anchor_score"] = anchors["anchor_score"] >= config.anchor_threshold
        anchors["is_strong_relation_survival"] = anchors["relation_survival_rate"] >= config.anchor_threshold
        anchors["is_strong_sector_survival"] = anchors["sector_relation_survival_rate"] >= config.anchor_threshold

        anchors = anchors.sort_values(
            ["anchor_score", "relation_survival_rate", "sector_relation_survival_rate"],
            ascending=[False, False, False],
        ).head(config.top_anchor_n)
        frames.append(anchors)

    return pd.concat(frames, ignore_index=True)


def build_cross_corpus_contrast(table: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "generator_completed_top1_survival_rate",
        "proto_edge_top1_survival_rate",
        "proto_relation_top1_survival_rate",
        "proto_sector_relation_top1_survival_rate",
        "gateway_event_top1_survival_rate",
        "gateway_event_relation_top1_survival_rate",
        "canonical_family_top1_survival_rate",
        "canonical_family_relation_top1_survival_rate",
        "anchor_canonical_family_top1_survival_rate",
        "gateway_relation_minus_proto_relation_delta",
        "canonical_relation_minus_proto_relation_delta",
        "anchor_family_minus_proto_relation_delta",
    ]

    rows = []
    for (band, route_class), group in table.groupby(["band", "replacement_route_class"]):
        by_corpus = {row["corpus_label"]: row for _, row in group.iterrows()}
        if "C" not in by_corpus or "Cp" not in by_corpus:
            continue
        row: dict[str, object] = {
            "band": band,
            "replacement_route_class": route_class,
            "C_projection_absorption_mode": by_corpus["C"].get("projection_absorption_mode"),
            "Cp_projection_absorption_mode": by_corpus["Cp"].get("projection_absorption_mode"),
            "same_projection_absorption_mode": by_corpus["C"].get("projection_absorption_mode")
            == by_corpus["Cp"].get("projection_absorption_mode"),
            "C_proto_sufficiency_mode": by_corpus["C"].get("proto_sufficiency_mode"),
            "Cp_proto_sufficiency_mode": by_corpus["Cp"].get("proto_sufficiency_mode"),
            "same_proto_sufficiency_mode": by_corpus["C"].get("proto_sufficiency_mode")
            == by_corpus["Cp"].get("proto_sufficiency_mode"),
        }
        for col in value_cols:
            c_val = safe_num(by_corpus["C"].get(col))
            cp_val = safe_num(by_corpus["Cp"].get(col))
            row[f"C_{col}"] = c_val
            row[f"Cp_{col}"] = cp_val
            row[f"Cp_minus_C_{col}"] = cp_val - c_val
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["band", "replacement_route_class"])


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def bullet_metric(label: str, value: object, digits: int = 3) -> str:
    return f"- {label}: {fmt_float(value, digits)}"


def mode_counts_lines(df: pd.DataFrame, mode_col: str) -> list[str]:
    counts = df[mode_col].value_counts(dropna=False).reset_index()
    counts.columns = [mode_col, "n"]
    return [f"- {row[mode_col]}: {int(row['n'])}" for _, row in counts.iterrows()]


def representative_rows(table: pd.DataFrame, mode: str, n: int = 3) -> pd.DataFrame:
    sub = table[table["projection_absorption_mode"] == mode].copy()
    if sub.empty:
        return sub
    return sub.sort_values(
        [
            "gateway_relation_minus_proto_relation_delta",
            "canonical_relation_minus_proto_relation_delta",
            "anchor_family_minus_proto_relation_delta",
        ],
        ascending=[False, False, False],
    ).head(n)


def write_report(
    config: StudyConfig,
    coupling_table: pd.DataFrame,
    projection_modes: pd.DataFrame,
    proto_sufficiency: pd.DataFrame,
    anchor_transfer: pd.DataFrame,
    cross_corpus: pd.DataFrame,
    diagnostics: dict[str, dict[str, int]],
) -> Path:
    report_path = config.out_dir / "obs067_proto_gateway_coupling_meta_report.md"
    lines: list[str] = []

    lines.append(f"# {STUDY_ID} — {STUDY_TITLE}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-067 tests how proto-groupoid survival from OBS-065 couples to gateway, "
        "canonical-family, and anchor-family survival from OBS-066 under the same "
        "route-origin decoy replacement regimes."
    )
    lines.append("")
    lines.append("It is a meta-analysis over compact summary artifacts. It does not rescan path-level symbolic traces.")
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    for corpus_label in config.corpora:
        paths = input_paths_for(config, corpus_label)
        lines.append(f"### {corpus_label}")
        lines.append("")
        for path in [
            paths.proto_summary,
            paths.proto_failure_modes,
            paths.proto_anchor_candidates,
            paths.gateway_summary,
            paths.canonical_summary,
            paths.anchor_summary,
            paths.consequence_modes,
        ]:
            lines.append(f"- `{rel(path, config.root)}`")
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- survival_high: `{config.survival_high}`")
    lines.append(f"- survival_low: `{config.survival_low}`")
    lines.append(f"- delta_threshold: `{config.delta_threshold}`")
    lines.append(f"- anchor_threshold: `{config.anchor_threshold}`")
    lines.append(f"- top_anchor_n: `{config.top_anchor_n}`")
    lines.append("")

    lines.append("## Diagnostic totals")
    lines.append("")
    lines.append(f"- coupling rows: {len(coupling_table)}")
    for corpus_label, vals in diagnostics.items():
        lines.append(f"- {corpus_label} joined regime rows: {vals['joined_regime_rows']}")
        lines.append(f"- {corpus_label} proto summary rows: {vals['proto_summary_rows']}")
        lines.append(f"- {corpus_label} gateway summary rows: {vals['gateway_summary_rows']}")
        lines.append(f"- {corpus_label} canonical summary rows: {vals['canonical_summary_rows']}")
        lines.append(f"- {corpus_label} anchor summary rows: {vals['anchor_summary_rows']}")
    lines.append("")

    lines.append("## Projection / absorption modes")
    lines.append("")
    for corpus_label in config.corpora:
        lines.append(f"### {corpus_label}")
        lines.append("")
        sub = coupling_table[coupling_table["corpus_label"] == corpus_label]
        lines.extend(mode_counts_lines(sub, "projection_absorption_mode"))
        lines.append("")

    lines.append("## Mean coupling by projection mode")
    lines.append("")
    for _, row in projection_modes.iterrows():
        lines.append(f"### {row['corpus_label']} — {row['projection_absorption_mode']}")
        lines.append("")
        lines.append(f"- n_regimes: {int(row['n_regimes'])}")
        lines.append(bullet_metric("mean_proto_relation_survival", row["mean_proto_relation_survival"]))
        lines.append(bullet_metric("mean_gateway_relation_survival", row["mean_gateway_relation_survival"]))
        lines.append(bullet_metric("mean_canonical_relation_survival", row["mean_canonical_relation_survival"]))
        lines.append(bullet_metric("mean_anchor_family_survival", row["mean_anchor_family_survival"]))
        lines.append(bullet_metric("mean_gateway_relation_minus_proto_delta", row["mean_gateway_relation_minus_proto_delta"]))
        lines.append(bullet_metric("mean_canonical_relation_minus_proto_delta", row["mean_canonical_relation_minus_proto_delta"]))
        lines.append("")

    lines.append("## Proto sufficiency modes")
    lines.append("")
    for corpus_label in config.corpora:
        lines.append(f"### {corpus_label}")
        lines.append("")
        sub = coupling_table[coupling_table["corpus_label"] == corpus_label]
        lines.extend(mode_counts_lines(sub, "proto_sufficiency_mode"))
        lines.append("")

    lines.append("## Anchor transfer modes")
    lines.append("")
    for corpus_label in config.corpora:
        lines.append(f"### {corpus_label}")
        lines.append("")
        sub = coupling_table[coupling_table["corpus_label"] == corpus_label]
        lines.extend(mode_counts_lines(sub, "anchor_transfer_mode"))
        lines.append("")

    lines.append("## Strongest retained anchor candidates")
    lines.append("")
    for corpus_label in config.corpora:
        sub = anchor_transfer[anchor_transfer["corpus_label"] == corpus_label].head(10)
        lines.append(f"### {corpus_label}")
        lines.append("")
        if sub.empty:
            lines.append("- none")
        else:
            for _, row in sub.iterrows():
                lines.append(
                    "- "
                    f"{row['band']} / replace {row['replacement_route_class']} / "
                    f"eval {row['evaluated_route_class']} / "
                    f"{row['baseline_proto_relation']} — "
                    f"relation_survival={fmt_float(row['relation_survival_rate'])}, "
                    f"sector_survival={fmt_float(row['sector_relation_survival_rate'])}, "
                    f"anchor_score={fmt_float(row['anchor_score'])}, "
                    f"mode={row.get('anchor_transfer_mode', 'NA')}"
                )
        lines.append("")

    lines.append("## Cross-corpus contrast")
    lines.append("")
    if cross_corpus.empty:
        lines.append("No paired C/Cp contrast rows were available.")
    else:
        same_projection = int(cross_corpus["same_projection_absorption_mode"].sum())
        same_sufficiency = int(cross_corpus["same_proto_sufficiency_mode"].sum())
        lines.append(f"- paired band × route rows: {len(cross_corpus)}")
        lines.append(f"- same projection_absorption_mode rows: {same_projection}")
        lines.append(f"- same proto_sufficiency_mode rows: {same_sufficiency}")
        lines.append("")
        lines.append("Largest absolute Cp-minus-C proto-relation differences:")
        diff_col = "Cp_minus_C_proto_relation_top1_survival_rate"
        top = cross_corpus.assign(abs_diff=cross_corpus[diff_col].abs()).sort_values("abs_diff", ascending=False).head(5)
        for _, row in top.iterrows():
            lines.append(
                "- "
                f"{row['band']} / replace {row['replacement_route_class']}: "
                f"Cp-C proto_relation={fmt_float(row[diff_col])}, "
                f"C_mode={row['C_projection_absorption_mode']}, "
                f"Cp_mode={row['Cp_projection_absorption_mode']}"
            )
    lines.append("")

    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append(
        "OBS-067 is a meta-analysis of OBS-065 and OBS-066 summary artifacts. "
        "It does not rerun route-origin decoys, does not recompute symbolic traces, "
        "and does not establish causal gateway mechanisms."
    )
    lines.append("")
    lines.append(
        "A downstream layer that survives while fine proto-relations fail should be "
        "read as projection or absorption of fine symbolic drift, not as proof that "
        "the fine proto-groupoid structure itself survived."
    )
    lines.append("")
    lines.append(
        "Top-1 survival remains sensitive to small baseline margins. OBS-067 therefore "
        "preserves top-3 overlap, TV distance, baseline margins, and inherited OBS-065/066 "
        "mode labels in the coupling table."
    )
    lines.append("")

    lines.append("## Result")
    lines.append("")
    lines.append(
        "OBS-067 should be read as the coupling ledger between proto-groupoid survival "
        "and downstream gateway/canonical-family survival. Its main result is the "
        "classification of each corpus × band × route-class regime as tracking, absorbing, "
        "amplifying, or decoupling from fine proto-groupoid drift."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    config = parse_args()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    coupling_table, diagnostics = build_layer_coupling_table(config)
    projection_modes = build_projection_absorption_modes(coupling_table)
    proto_sufficiency = build_proto_sufficiency_table(coupling_table)
    anchor_transfer = build_anchor_transfer_table(config, coupling_table)
    cross_corpus = build_cross_corpus_contrast(coupling_table)

    coupling_path = config.out_dir / "obs067_layer_coupling_table.csv"
    projection_path = config.out_dir / "obs067_projection_absorption_modes.csv"
    sufficiency_path = config.out_dir / "obs067_proto_sufficiency_table.csv"
    anchor_path = config.out_dir / "obs067_anchor_transfer_table.csv"
    contrast_path = config.out_dir / "obs067_cross_corpus_contrast.csv"

    coupling_table.to_csv(coupling_path, index=False)
    projection_modes.to_csv(projection_path, index=False)
    proto_sufficiency.to_csv(sufficiency_path, index=False)
    anchor_transfer.to_csv(anchor_path, index=False)
    cross_corpus.to_csv(contrast_path, index=False)

    report_path = write_report(
        config=config,
        coupling_table=coupling_table,
        projection_modes=projection_modes,
        proto_sufficiency=proto_sufficiency,
        anchor_transfer=anchor_transfer,
        cross_corpus=cross_corpus,
        diagnostics=diagnostics,
    )

    print(f"{STUDY_ID} complete")
    print(f"wrote: {rel(coupling_path, config.root)}")
    print(f"wrote: {rel(projection_path, config.root)}")
    print(f"wrote: {rel(sufficiency_path, config.root)}")
    print(f"wrote: {rel(anchor_path, config.root)}")
    print(f"wrote: {rel(contrast_path, config.root)}")
    print(f"wrote: {rel(report_path, config.root)}")


if __name__ == "__main__":
    main()

