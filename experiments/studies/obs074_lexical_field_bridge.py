#!/usr/bin/env python3
"""
OBS-074 — Lexical substrate / field-geometry bridge, v3.

Purpose
-------
Test whether corpus/path observatory differences are explainable by lexical
surface statistics, or whether continuous-field geometry retains signal after
lexical controls.

v3 patch
--------
v2 fixed lexical join semantics by separating:

  - path-level lexical joins, valid only when path_id overlap exists
  - corpus-level lexical fingerprints, valid as corpus-regime controls

v3 adds OBS-073-style field blinding:

  - field_only
  - field_no_direct_seam
  - field_no_grid
  - field_no_direct_seam_no_grid

and crosses those with corpus lexical controls:

  - corpus_lexical_only
  - corpus_lexical_plus_field

Interpretation discipline
-------------------------
- Corpus-level lexical controls can behave as corpus-regime proxies.
- Path-level lexical controls are only valid when lexical_path_overlap_share is nontrivial.
- Field signal surviving no_direct_seam_no_grid is stronger than unblinded field signal.
- Lexical predictability does not falsify field geometry; it identifies a surface confound.
- Tokenizer-specific claims require tokenizer-aligned analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusSpec:
    label: str
    root: Path
    text_source: Path | None = None
    scale: str = "100000"

    @property
    def family_substrate_dir(self) -> Path:
        return self.root / "scales" / self.scale / "family_substrate"

    @property
    def path_node_diagnostics_csv(self) -> Path:
        return self.family_substrate_dir / "path_node_diagnostics.csv"

    @property
    def path_diagnostics_csv(self) -> Path:
        return self.family_substrate_dir / "path_diagnostics.csv"

    @property
    def family_assignments_csv(self) -> Path:
        return self.family_substrate_dir / "path_family_assignments.csv"

    @property
    def obs050_segments_csv(self) -> Path:
        return self.root / "obs050_structural_coupling_persistence" / "structural_coupling_segments.csv"


@dataclass(frozen=True)
class Config:
    corpora: list[CorpusSpec]
    outdir: Path
    random_state: int = 74
    min_class_count: int = 20
    max_rows_per_corpus: int | None = None
    rf_n_estimators: int = 500
    rf_max_depth: int = 5
    rf_min_samples_leaf: int = 20
    permutation_repeats: int = 8
    n_top_importances: int = 10


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target_col: str


TARGET_SPECS = [
    TargetSpec("path_family", "target_path_family"),
    TargetSpec("coupling_class", "target_coupling_class"),
    TargetSpec("outcome_group", "target_outcome_group"),
    TargetSpec("coupled_outcome_group", "target_coupled_outcome_group"),
    TargetSpec("recovery_channel_structural", "target_recovery_channel_structural"),
]


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    include_field: bool
    include_lexical: bool
    lexical_scope: str
    no_direct_seam: bool = False
    no_grid_location: bool = False


FIELD_VARIANTS = [
    FeatureSetSpec(
        "field_only",
        include_field=True,
        include_lexical=False,
        lexical_scope="none",
    ),
    FeatureSetSpec(
        "field_no_direct_seam",
        include_field=True,
        include_lexical=False,
        lexical_scope="none",
        no_direct_seam=True,
    ),
    FeatureSetSpec(
        "field_no_grid",
        include_field=True,
        include_lexical=False,
        lexical_scope="none",
        no_grid_location=True,
    ),
    FeatureSetSpec(
        "field_no_direct_seam_no_grid",
        include_field=True,
        include_lexical=False,
        lexical_scope="none",
        no_direct_seam=True,
        no_grid_location=True,
    ),
]

LEXICAL_VARIANTS = [
    FeatureSetSpec(
        "corpus_lexical_only",
        include_field=False,
        include_lexical=True,
        lexical_scope="corpus",
    ),
    FeatureSetSpec(
        "path_lexical_only",
        include_field=False,
        include_lexical=True,
        lexical_scope="path",
    ),
]

COMBINED_VARIANTS = [
    FeatureSetSpec(
        "corpus_lexical_plus_field",
        include_field=True,
        include_lexical=True,
        lexical_scope="corpus",
    ),
    FeatureSetSpec(
        "corpus_lexical_plus_field_no_direct_seam",
        include_field=True,
        include_lexical=True,
        lexical_scope="corpus",
        no_direct_seam=True,
    ),
    FeatureSetSpec(
        "corpus_lexical_plus_field_no_grid",
        include_field=True,
        include_lexical=True,
        lexical_scope="corpus",
        no_grid_location=True,
    ),
    FeatureSetSpec(
        "corpus_lexical_plus_field_no_direct_seam_no_grid",
        include_field=True,
        include_lexical=True,
        lexical_scope="corpus",
        no_direct_seam=True,
        no_grid_location=True,
    ),
    FeatureSetSpec(
        "path_lexical_plus_field",
        include_field=True,
        include_lexical=True,
        lexical_scope="path",
    ),
    FeatureSetSpec(
        "path_lexical_plus_field_no_direct_seam",
        include_field=True,
        include_lexical=True,
        lexical_scope="path",
        no_direct_seam=True,
    ),
    FeatureSetSpec(
        "path_lexical_plus_field_no_grid",
        include_field=True,
        include_lexical=True,
        lexical_scope="path",
        no_grid_location=True,
    ),
    FeatureSetSpec(
        "path_lexical_plus_field_no_direct_seam_no_grid",
        include_field=True,
        include_lexical=True,
        lexical_scope="path",
        no_direct_seam=True,
        no_grid_location=True,
    ),
]

FEATURE_SET_SPECS = FIELD_VARIANTS + LEXICAL_VARIANTS + COMBINED_VARIANTS


# ---------------------------------------------------------------------
# Leakage / feature provenance rules
# ---------------------------------------------------------------------


TARGET_OR_METADATA_EXACT = {
    "path_id",
    "probe_id",
    "corpus",
    "source_root",
    "scale",
    "path_family",
    "target_path_family",
    "target_coupling_class",
    "target_outcome_group",
    "target_coupled_outcome_group",
    "target_recovery_channel_structural",
}

OBS050_METADATA_PREFIXES = ("obs050_",)

DIRECT_SEAM_PATTERNS = [
    r"distance_to_seam",
    r"(^|_)seam($|_)",
    r"near_fraction",
    r"mid_fraction",
    r"far_fraction",
    r"core_fraction",
    r"coupled",
    r"coupling",
    r"m_seam",
    r"min_distance",
    r"mean_distance",
]

SYMBOLIC_PATTERNS = [
    r"family$",
    r"path_family",
    r"outcome",
    r"posture",
    r"seam_band",
    r"coupling_class",
]

GRID_LOCATION_PATTERNS = [
    r"^field_pn_node_id_x_(mean|std|min|max|median|sum)$",
    r"^field_pn_node_id_y_(mean|std|min|max|median|sum)$",
    r"^field_pn_r_(mean|std|min|max|median|sum)$",
    r"^field_pn_alpha_(mean|std|min|max|median|sum)$",
    r"^field_pn_mds1_(mean|std|min|max|median|sum)$",
    r"^field_pn_mds2_(mean|std|min|max|median|sum)$",
    r"^field_pd_start_",
    r"^field_pd_end_",
    r"^field_pd_initial_",
    r"^field_pd_final_",
    r"^field_pd_mean_r$",
    r"^field_pd_mean_alpha$",
    r"^field_pd_min_r$",
    r"^field_pd_max_r$",
    r"^field_pd_min_alpha$",
    r"^field_pd_max_alpha$",
]


def matches_any(col: str, patterns: list[str]) -> bool:
    return any(re.search(p, col, flags=re.IGNORECASE) for p in patterns)


def classify_feature(col: str) -> str:
    if col in TARGET_OR_METADATA_EXACT or col.startswith("target_"):
        return "target_or_metadata"
    if col.startswith(OBS050_METADATA_PREFIXES):
        return "obs050_metadata"
    if matches_any(col, SYMBOLIC_PATTERNS):
        return "symbolic"
    if col.startswith("lex_"):
        return "lexical"
    if col.startswith("field_"):
        if matches_any(col, DIRECT_SEAM_PATTERNS):
            return "direct_seam"
        if matches_any(col, GRID_LOCATION_PATTERNS):
            return "grid_location"
        return "field"
    return "other"


def feature_allowed(col: str, fs: FeatureSetSpec) -> tuple[bool, str]:
    prov = classify_feature(col)

    if prov in {"target_or_metadata", "obs050_metadata", "symbolic"}:
        return False, prov

    if col.startswith("field_"):
        if not fs.include_field:
            return False, "field_excluded"
        if fs.no_direct_seam and prov == "direct_seam":
            return False, "direct_seam_blinded"
        if fs.no_grid_location and prov == "grid_location":
            return False, "grid_location_blinded"
        return True, prov

    if col.startswith("lex_"):
        if not fs.include_lexical:
            return False, "lexical_excluded"
        if fs.lexical_scope == "corpus" and not col.startswith("lex_corpus_"):
            return False, "path_lexical_excluded"
        if fs.lexical_scope == "path" and not col.startswith("lex_path_"):
            return False, "corpus_lexical_excluded"
        if fs.lexical_scope == "none":
            return False, "lexical_excluded"
        return True, "lexical"

    return False, prov


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required {label} exists but is empty: {path}")
    return pd.read_csv(path)


def normalize_path_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "path_id" not in out.columns and "probe_id" in out.columns:
        out = out.rename(columns={"probe_id": "path_id"})
    if "path_id" in out.columns:
        out["path_id"] = out["path_id"].astype(str)
    return out


def normalize_outcome_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outcome_group" not in out.columns and "outcome" in out.columns:
        out = out.rename(columns={"outcome": "outcome_group"})
    return out


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def finite_mean(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.mean()) if y.notna().any() else float("nan")


def finite_std(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.std()) if y.notna().sum() > 1 else float("nan")


def finite_min(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.min()) if y.notna().any() else float("nan")


def finite_max(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.max()) if y.notna().any() else float("nan")


def finite_median(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.median()) if y.notna().any() else float("nan")


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


# ---------------------------------------------------------------------
# Lexical extraction
# ---------------------------------------------------------------------


MODAL_WORDS = {
    "may",
    "might",
    "could",
    "would",
    "should",
    "can",
    "cannot",
    "likely",
    "perhaps",
    "possibly",
    "probably",
}

MECHANISTIC_WORDS = {
    "because",
    "therefore",
    "hence",
    "so",
    "mechanism",
    "process",
    "structure",
    "operator",
    "field",
    "geometry",
    "constraint",
    "transition",
    "coupling",
    "trajectory",
    "system",
    "dynamics",
    "signal",
    "phase",
    "seam",
}

CAUSAL_WORDS = {
    "because",
    "therefore",
    "thus",
    "hence",
    "since",
    "so",
    "causes",
    "caused",
    "drives",
    "implies",
    "leads",
}

UNCERTAINTY_WORDS = {
    "maybe",
    "perhaps",
    "possibly",
    "likely",
    "unlikely",
    "unclear",
    "uncertain",
    "unknown",
    "ambiguous",
    "provisional",
}

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def safe_div(a: float, b: float) -> float:
    if not b:
        return 0.0
    return float(a / b)


def trigram_share(tokens: list[str]) -> float:
    if len(tokens) < 3:
        return 0.0
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    counts = Counter(trigrams)
    if not counts:
        return 0.0
    return float(max(counts.values()) / len(trigrams))


def lexical_features_for_text(text: str) -> dict[str, float]:
    text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    tokens = tokenize(text)
    n_words = len(tokens)
    n_chars = len(text)

    sentences = re.split(r"[.!?]+", text)
    sentence_lengths = [len(tokenize(s)) for s in sentences if tokenize(s)]

    modal_count = sum(t in MODAL_WORDS for t in tokens)
    mechanistic_count = sum(t in MECHANISTIC_WORDS for t in tokens)
    causal_count = sum(t in CAUSAL_WORDS for t in tokens)
    uncertainty_count = sum(t in UNCERTAINTY_WORDS for t in tokens)

    return {
        "word_count": float(n_words),
        "char_count": float(n_chars),
        "type_token_ratio": safe_div(len(set(tokens)), n_words),
        "avg_word_length": safe_div(sum(len(t) for t in tokens), n_words),
        "newline_density": safe_div(text.count("\n"), max(n_chars, 1)),
        "markdown_bold_density": safe_div(text.count("**"), max(n_chars, 1)),
        "bullet_density": safe_div(text.count("- "), max(n_chars, 1)),
        "colon_density": safe_div(text.count(":"), max(n_chars, 1)),
        "semicolon_density": safe_div(text.count(";"), max(n_chars, 1)),
        "question_mark_density": safe_div(text.count("?"), max(n_chars, 1)),
        "modal_density": safe_div(modal_count, n_words),
        "mechanistic_density": safe_div(mechanistic_count, n_words),
        "causal_density": safe_div(causal_count, n_words),
        "uncertainty_density": safe_div(uncertainty_count, n_words),
        "top_trigram_share": trigram_share(tokens),
        "sentence_count": float(len(sentence_lengths)),
        "mean_sentence_words": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "max_sentence_words": float(np.max(sentence_lengths)) if sentence_lengths else 0.0,
        "empty_response": float(n_chars == 0),
        "cutoff_like": float(
            n_chars > 0
            and not text.rstrip().endswith((".", "!", "?", ")", "]", '"', "'", "`"))
        ),
    }


def load_text_rows(path: Path | None) -> tuple[pd.DataFrame, str]:
    if path is None:
        return pd.DataFrame(), "missing_text_source"
    if not path.exists():
        return pd.DataFrame(), "missing"

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return pd.DataFrame(), f"json_error:{exc}"

    rows: list[dict[str, Any]] = []

    def add_row(i: int, item: Any) -> None:
        if isinstance(item, str):
            rows.append({"text_row_id": str(i), "text": item})
            return

        if isinstance(item, dict):
            text = (
                item.get("response")
                or item.get("text")
                or item.get("content")
                or item.get("answer")
                or item.get("completion")
                or ""
            )
            path_id = item.get("path_id") or item.get("probe_id") or item.get("id")
            step = item.get("step") or item.get("turn") or item.get("response_index")
            row = {
                "text_row_id": str(i),
                "text": text,
            }
            if path_id is not None:
                row["path_id"] = str(path_id)
            if step is not None:
                row["text_step"] = step
            rows.append(row)

    if isinstance(raw, list):
        for i, item in enumerate(raw):
            add_row(i, item)
    elif isinstance(raw, dict):
        candidate = None
        for key in ["responses", "items", "records", "data", "messages"]:
            if isinstance(raw.get(key), list):
                candidate = raw[key]
                break
        if candidate is not None:
            for i, item in enumerate(candidate):
                add_row(i, item)
        else:
            add_row(0, raw)
    else:
        return pd.DataFrame(), "unsupported_json_shape"

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "empty"
    return df, "ok"


def build_lexical_tables(corpus: CorpusSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    text_rows, status = load_text_rows(corpus.text_source)

    manifest = {
        "corpus": corpus.label,
        "root": str(corpus.root),
        "text_source": str(corpus.text_source) if corpus.text_source is not None else "",
        "text_load_status": status,
        "n_text_rows": int(len(text_rows)),
    }

    if text_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), manifest

    feat_rows = []
    for _, row in text_rows.iterrows():
        feats = lexical_features_for_text(str(row.get("text", "")))
        out = {
            "corpus": corpus.label,
            "text_row_id": row.get("text_row_id", ""),
        }
        if "path_id" in row:
            out["path_id"] = str(row["path_id"])
        out.update({f"lex_path_{k}": v for k, v in feats.items()})
        feat_rows.append(out)

    path_lex = pd.DataFrame(feat_rows)

    numeric_cols = [c for c in path_lex.columns if c.startswith("lex_path_")]
    corpus_row = {"corpus": corpus.label}
    for col in numeric_cols:
        base = col.replace("lex_path_", "lex_corpus_")
        s = pd.to_numeric(path_lex[col], errors="coerce")
        corpus_row[f"{base}_mean"] = finite_mean(s)
        corpus_row[f"{base}_std"] = finite_std(s)
        corpus_row[f"{base}_min"] = finite_min(s)
        corpus_row[f"{base}_max"] = finite_max(s)
        corpus_row[f"{base}_median"] = finite_median(s)

    corpus_lex = pd.DataFrame([corpus_row])
    return path_lex, corpus_lex, manifest


# ---------------------------------------------------------------------
# Field feature construction
# ---------------------------------------------------------------------


def summarize_numeric_by_path(path_nodes: pd.DataFrame) -> pd.DataFrame:
    ignore = {
        "path_id",
        "probe_id",
        "step",
        "node_id",
        "path_family",
        "outcome_group",
        "coupling_class",
        "seam_band",
        "posture",
    }

    numeric_cols = []
    for c in path_nodes.columns:
        if c in ignore:
            continue
        s = pd.to_numeric(path_nodes[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)

    rows = []
    for path_id, grp in path_nodes.groupby("path_id", sort=False):
        row: dict[str, Any] = {"path_id": str(path_id)}
        for col in numeric_cols:
            s = pd.to_numeric(grp[col], errors="coerce")
            if not s.notna().any():
                continue

            prefix = f"field_pn_{col}"
            row[f"{prefix}_mean"] = finite_mean(s)
            row[f"{prefix}_std"] = finite_std(s)
            row[f"{prefix}_min"] = finite_min(s)
            row[f"{prefix}_max"] = finite_max(s)
            row[f"{prefix}_median"] = finite_median(s)
            row[f"{prefix}_sum"] = float(s.sum(skipna=True))

            nz = s.dropna()
            row[f"{prefix}_last_minus_first"] = (
                float(nz.iloc[-1] - nz.iloc[0]) if len(nz) >= 2 else np.nan
            )

        row["field_pn_n_steps"] = int(len(grp))
        rows.append(row)

    return pd.DataFrame(rows)


def reduce_path_diagnostics(path_diag: pd.DataFrame) -> pd.DataFrame:
    path_diag = normalize_path_id(path_diag)
    out = path_diag.drop_duplicates("path_id").copy()

    keep = ["path_id"]
    for c in out.columns:
        if c == "path_id" or c == "path_family":
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().any():
            keep.append(c)

    out = out[keep].copy()
    return out.rename(columns={c: f"field_pd_{c}" for c in out.columns if c != "path_id"})


def normalize_obs050_classes(obs050: pd.DataFrame) -> pd.DataFrame:
    df = normalize_path_id(normalize_outcome_column(obs050))

    if "seam_band" not in df.columns:
        mean_d = numeric_series(df, "mean_distance_to_seam")
        min_d = numeric_series(df, "min_distance_to_seam")
        df["seam_band"] = np.where(
            min_d <= 1e-12,
            "core",
            np.where(mean_d <= 0.15, "near", "far"),
        )

    if "coupling_class" not in df.columns:
        df["coupling_class"] = np.where(
            df["seam_band"].astype(str).isin(["core", "near"]),
            "coupled",
            np.where(df["seam_band"].astype(str).eq("far"), "decoupled", "unknown"),
        )

    if "posture" not in df.columns:
        m = numeric_series(df, "m_seam")
        df["posture"] = np.where(
            m < -0.02,
            "compression",
            np.where(m > 0.02, "dissipation", "graze"),
        )

    return df


def derive_obs050_targets(obs050: pd.DataFrame) -> pd.DataFrame:
    df = normalize_obs050_classes(obs050)

    rows = []
    for path_id, grp in df.groupby("path_id", sort=False):
        outcome_mode = (
            grp["outcome_group"].astype(str).mode().iloc[0]
            if "outcome_group" in grp.columns and not grp.empty
            else pd.NA
        )

        coupled_mask = grp["coupling_class"].astype(str).eq("coupled")
        coupled_share = float(coupled_mask.mean()) if len(grp) else 0.0
        has_coupled = bool(coupled_mask.any())

        fam_col = (
            grp["path_family"].astype(str)
            if "path_family" in grp.columns
            else pd.Series("", index=grp.index)
        )

        false_mask = (
            grp["outcome_group"].astype(str).eq("nonrecovering")
            & coupled_mask
            & fam_col.eq("off_seam_reorganizing")
            & grp["seam_band"].astype(str).eq("near")
            & grp["posture"].astype(str).eq("compression")
        )

        true_mask = grp["outcome_group"].astype(str).eq("recovering") & coupled_mask

        if bool(false_mask.any()):
            channel = "false_recovery_compression"
        elif bool(true_mask.any()):
            channel = "true_bounded_recovery"
        else:
            channel = pd.NA

        rows.append(
            {
                "path_id": str(path_id),
                "target_outcome_group": outcome_mode,
                "target_coupling_class": "coupled" if coupled_share >= 0.5 or has_coupled else "decoupled",
                "target_coupled_outcome_group": outcome_mode if has_coupled else pd.NA,
                "target_recovery_channel_structural": channel,
                "obs050_n_segments": int(len(grp)),
                "obs050_coupled_share": coupled_share,
                "obs050_has_nonrecovering_coupled": int(bool(false_mask.any())),
                "obs050_mean_m_seam": finite_mean(grp.get("m_seam", pd.Series(dtype=float))),
                "obs050_mean_distance_to_seam": finite_mean(
                    grp.get("mean_distance_to_seam", pd.Series(dtype=float))
                ),
                "obs050_min_distance_to_seam": finite_min(
                    grp.get("min_distance_to_seam", pd.Series(dtype=float))
                ),
                "obs050_mean_roughness": finite_mean(
                    grp.get("mean_roughness", pd.Series(dtype=float))
                ),
            }
        )

    return pd.DataFrame(rows)


def build_feature_table_for_corpus(corpus: CorpusSpec) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    path_nodes = normalize_path_id(
        read_csv_required(corpus.path_node_diagnostics_csv, f"{corpus.label} path_node_diagnostics")
    )
    path_diag = normalize_path_id(
        read_csv_required(corpus.path_diagnostics_csv, f"{corpus.label} path_diagnostics")
    )
    family = normalize_path_id(
        read_csv_required(corpus.family_assignments_csv, f"{corpus.label} path_family_assignments")
    )
    obs050 = normalize_path_id(
        normalize_outcome_column(
            read_csv_required(corpus.obs050_segments_csv, f"{corpus.label} OBS-050 segments")
        )
    )

    field_nodes = summarize_numeric_by_path(path_nodes)
    field_diag = reduce_path_diagnostics(path_diag)
    targets = derive_obs050_targets(obs050)

    feature = field_nodes.merge(field_diag, on="path_id", how="left")

    if "path_family" in family.columns:
        feature = feature.merge(
            family[["path_id", "path_family"]].drop_duplicates("path_id"),
            on="path_id",
            how="left",
        )
    else:
        feature["path_family"] = pd.NA

    feature["target_path_family"] = feature["path_family"]
    feature = feature.merge(targets, on="path_id", how="left")

    path_lex, corpus_lex, lex_manifest = build_lexical_tables(corpus)

    feature["corpus"] = corpus.label
    feature["source_root"] = str(corpus.root)
    feature["scale"] = corpus.scale

    feature = feature.merge(corpus_lex, on="corpus", how="left") if not corpus_lex.empty else feature

    if not path_lex.empty and "path_id" in path_lex.columns:
        path_lex_unique = path_lex.drop_duplicates("path_id").copy()
        feature = feature.merge(path_lex_unique.drop(columns=["corpus"], errors="ignore"), on="path_id", how="left")

    manifest = {
        "corpus": corpus.label,
        "root": str(corpus.root),
        "scale": corpus.scale,
        "n_paths": int(feature["path_id"].nunique()),
        **lex_manifest,
    }

    return feature, manifest, path_lex, corpus_lex


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------


def make_rf(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def get_feature_columns(
    df: pd.DataFrame,
    fs: FeatureSetSpec,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    features = []

    for col in df.columns:
        allowed, reason = feature_allowed(col, fs)
        numeric_ok = False

        if allowed:
            s = pd.to_numeric(df[col], errors="coerce")
            numeric_ok = bool(s.notna().any())
            allowed = numeric_ok

        rows.append(
            {
                "feature_set": fs.name,
                "feature": col,
                "allowed": int(allowed),
                "provenance_or_exclusion": reason,
                "numeric_ok": int(numeric_ok),
                "include_field": int(fs.include_field),
                "include_lexical": int(fs.include_lexical),
                "lexical_scope": fs.lexical_scope,
                "no_direct_seam": int(fs.no_direct_seam),
                "no_grid_location": int(fs.no_grid_location),
            }
        )

        if allowed:
            features.append(col)

    return features, pd.DataFrame(rows)


def prepare_xy(
    df: pd.DataFrame,
    target: TargetSpec,
    fs: FeatureSetSpec,
    min_class_count: int,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    if target.target_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=str), [], pd.DataFrame()

    work = df[df[target.target_col].notna()].copy()
    work[target.target_col] = work[target.target_col].astype(str)

    counts = work[target.target_col].value_counts()
    valid_classes = counts[counts >= min_class_count].index.tolist()
    work = work[work[target.target_col].isin(valid_classes)].copy()

    feature_cols, feature_manifest = get_feature_columns(work, fs)
    feature_manifest["target"] = target.name

    X = work[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=work.index)
    X = X.replace([np.inf, -np.inf], np.nan)

    med = X.median(axis=0, skipna=True) if not X.empty else pd.Series(dtype=float)
    X = X.fillna(med).fillna(0.0)

    y = work[target.target_col].astype(str)
    return X, y, feature_cols, feature_manifest


def evaluate_model(
    cfg: Config,
    df: pd.DataFrame,
    target: TargetSpec,
    fs: FeatureSetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y, feature_cols, feature_manifest = prepare_xy(df, target, fs, cfg.min_class_count)

    score_rows: list[dict[str, Any]] = []
    perm_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    base_score = {
        "target": target.name,
        "target_col": target.target_col,
        "feature_set": fs.name,
        "include_field": int(fs.include_field),
        "include_lexical": int(fs.include_lexical),
        "lexical_scope": fs.lexical_scope,
        "no_direct_seam": int(fs.no_direct_seam),
        "no_grid_location": int(fs.no_grid_location),
    }

    if len(y.unique()) < 2 or len(X) < cfg.min_class_count * 2 or len(feature_cols) == 0:
        score_rows.append(
            {
                **base_score,
                "status": "insufficient_classes_rows_or_features",
                "n_rows": int(len(X)),
                "n_classes": int(y.nunique()),
                "feature_count": int(len(feature_cols)),
            }
        )
        return pd.DataFrame(score_rows), pd.DataFrame(perm_rows), pd.DataFrame(confusion_rows), feature_manifest

    min_count = int(y.value_counts().min())
    n_splits = max(2, min(5, min_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    rf = make_rf(cfg)
    y_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)

    score_rows.append(
        {
            **base_score,
            "status": "ok",
            "n_rows": int(len(X)),
            "n_classes": int(y.nunique()),
            "classes": json.dumps(sorted(y.unique().tolist())),
            "feature_count": int(len(feature_cols)),
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro"),
            "weighted_f1": f1_score(y, y_pred, average="weighted"),
        }
    )

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y, y_pred, labels=labels)
    for i, actual in enumerate(labels):
        for j, pred in enumerate(labels):
            confusion_rows.append(
                {
                    **base_score,
                    "actual": actual,
                    "predicted": pred,
                    "n": int(cm[i, j]),
                }
            )

    rf.fit(X, y)

    try:
        perm = permutation_importance(
            rf,
            X,
            y,
            n_repeats=cfg.permutation_repeats,
            random_state=cfg.random_state,
            n_jobs=-1,
            scoring="balanced_accuracy",
        )
        order = np.argsort(perm.importances_mean)[::-1]
        for rank, idx in enumerate(order, start=1):
            feat = feature_cols[idx]
            perm_rows.append(
                {
                    **base_score,
                    "rank": rank,
                    "feature": feat,
                    "importance_mean": float(perm.importances_mean[idx]),
                    "importance_std": float(perm.importances_std[idx]),
                    "feature_provenance": classify_feature(feat),
                }
            )
    except Exception as exc:
        warnings.warn(f"Permutation importance failed for {target.name}/{fs.name}: {exc}")

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(perm_rows),
        pd.DataFrame(confusion_rows),
        feature_manifest,
    )


# ---------------------------------------------------------------------
# Audit / reporting
# ---------------------------------------------------------------------


def build_lexical_join_audit(feature_table: pd.DataFrame) -> pd.DataFrame:
    path_cols = [c for c in feature_table.columns if c.startswith("lex_path_")]
    if not path_cols:
        return pd.DataFrame(
            [
                {
                    "lexical_path_join_mode": "none",
                    "lexical_path_overlap_rows": 0,
                    "lexical_path_overlap_share": 0.0,
                    "lexical_path_rows": 0,
                }
            ]
        )

    any_path_lex = feature_table[path_cols].notna().any(axis=1)
    return pd.DataFrame(
        [
            {
                "lexical_path_join_mode": "path_id",
                "lexical_path_overlap_rows": int(any_path_lex.sum()),
                "lexical_path_overlap_share": float(any_path_lex.mean()) if len(feature_table) else 0.0,
                "lexical_path_rows": int(any_path_lex.sum()),
            }
        ]
    )


def corpus_fingerprint(feature_table: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in feature_table.columns if c.startswith("lex_corpus_")]
    rows = []
    for corpus, grp in feature_table.groupby("corpus", sort=False):
        row = {
            "corpus": corpus,
            "n_paths_with_lexical_features": int(len(grp)),
        }
        for col in cols:
            s = pd.to_numeric(grp[col], errors="coerce")
            if s.notna().any():
                row[col] = finite_mean(s)
        rows.append(row)
    return pd.DataFrame(rows)


def build_target_manifest(feature_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_SPECS:
        if target.target_col not in feature_table.columns:
            continue
        tmp = feature_table[feature_table[target.target_col].notna()].copy()
        for corpus, grp in tmp.groupby("corpus", sort=False):
            counts = grp[target.target_col].astype(str).value_counts()
            for cls, n in counts.items():
                rows.append(
                    {
                        "target": target.name,
                        "target_col": target.target_col,
                        "corpus": corpus,
                        "class": cls,
                        "n": int(n),
                    }
                )
    return pd.DataFrame(rows)


def lexical_vs_field_read_table(model_scores: pd.DataFrame) -> pd.DataFrame:
    ok = model_scores[model_scores["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()

    def ba(target: str, feature_set: str) -> float:
        s = ok[(ok["target"].eq(target)) & (ok["feature_set"].eq(feature_set))]["balanced_accuracy"]
        return float(pd.to_numeric(s, errors="coerce").max()) if len(s) else float("nan")

    rows = []
    for target in [t.name for t in TARGET_SPECS]:
        field = ba(target, "field_only")
        blind = ba(target, "field_no_direct_seam_no_grid")
        lex = ba(target, "corpus_lexical_only")
        plus = ba(target, "corpus_lexical_plus_field")
        plus_blind = ba(target, "corpus_lexical_plus_field_no_direct_seam_no_grid")
        path_lex = ba(target, "path_lexical_only")
        path_plus_blind = ba(target, "path_lexical_plus_field_no_direct_seam_no_grid")

        rows.append(
            {
                "target": target,
                "field_only_ba": field,
                "field_no_direct_seam_no_grid_ba": blind,
                "corpus_lexical_only_ba": lex,
                "corpus_lexical_plus_field_ba": plus,
                "corpus_lexical_plus_field_no_direct_seam_no_grid_ba": plus_blind,
                "delta_plus_field_minus_corpus_lexical": plus - lex if np.isfinite(plus) and np.isfinite(lex) else np.nan,
                "delta_blinded_plus_field_minus_corpus_lexical": plus_blind - lex if np.isfinite(plus_blind) and np.isfinite(lex) else np.nan,
                "delta_blinded_field_minus_corpus_lexical": blind - lex if np.isfinite(blind) and np.isfinite(lex) else np.nan,
                "path_lexical_only_ba": path_lex,
                "path_lexical_plus_field_no_direct_seam_no_grid_ba": path_plus_blind,
            }
        )

    return pd.DataFrame(rows)


def write_summary(
    cfg: Config,
    manifests: list[dict[str, Any]],
    feature_table: pd.DataFrame,
    lexical_audit: pd.DataFrame,
    lexical_fingerprint: pd.DataFrame,
    model_scores: pd.DataFrame,
    read_table: pd.DataFrame,
    perm: pd.DataFrame,
    feature_manifest: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# OBS-074 — Lexical substrate / field-geometry bridge v3",
        "",
        "## Scope",
        "",
        "OBS-074 tests whether corpus/path observatory differences are explainable by lexical surface statistics,",
        "or whether continuous-field geometry retains signal after lexical controls.",
        "",
        "v3 adds OBS-073-style direct-seam and grid-location blinding to the lexical bridge.",
        "",
        "This report is artifact-first. It does not recompute geometry, path families, coupling, or divergence.",
        "",
        "## Inputs",
        "",
        "| corpus | root | text_source | text_load_status | n_paths | n_text_rows |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]

    for m in manifests:
        lines.append(
            f"| {m['corpus']} | `{m['root']}` | `{m.get('text_source', '')}` | "
            f"{m.get('text_load_status', '')} | {m.get('n_paths', 0)} | {m.get('n_text_rows', 0)} |"
        )

    lines.extend(["", "## Lexical path join audit", ""])
    if lexical_audit.empty:
        lines.append("No lexical audit available.")
    else:
        cols = lexical_audit.columns.tolist()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in lexical_audit.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    lines.extend(["", "## Lexical corpus fingerprint", ""])
    if lexical_fingerprint.empty:
        lines.append("No lexical corpus fingerprint available.")
    else:
        display_cols = [
            "corpus",
            "n_paths_with_lexical_features",
            "lex_corpus_word_count_mean",
            "lex_corpus_char_count_mean",
            "lex_corpus_type_token_ratio_mean",
            "lex_corpus_newline_density_mean",
            "lex_corpus_markdown_bold_density_mean",
            "lex_corpus_modal_density_mean",
            "lex_corpus_mechanistic_density_mean",
            "lex_corpus_top_trigram_share_mean",
            "lex_corpus_empty_response_mean",
            "lex_corpus_cutoff_like_mean",
        ]
        use_cols = [c for c in display_cols if c in lexical_fingerprint.columns]
        lines.append("| " + " | ".join(use_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_cols)) + " |")
        for _, row in lexical_fingerprint.iterrows():
            vals = []
            for c in use_cols:
                vals.append(fmt(row[c], 6) if c != "corpus" else str(row[c]))
            lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## Ablation scores", ""])
    ok = model_scores[model_scores["status"].eq("ok")].copy() if not model_scores.empty else pd.DataFrame()
    if ok.empty:
        lines.append("No successful model runs.")
    else:
        cols = [
            "target",
            "feature_set",
            "n_rows",
            "n_classes",
            "feature_count",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "no_direct_seam",
            "no_grid_location",
            "lexical_scope",
        ]
        use_cols = [c for c in cols if c in ok.columns]
        lines.append("| " + " | ".join(use_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_cols)) + " |")
        for _, row in ok.sort_values(["target", "feature_set"]).iterrows():
            vals = []
            for c in use_cols:
                val = row[c]
                vals.append(fmt(val, 4) if isinstance(val, (float, np.floating)) else str(val))
            lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## Lexical vs field read", ""])
    if read_table.empty:
        lines.append("No lexical-vs-field read table available.")
    else:
        cols = read_table.columns.tolist()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in read_table.iterrows():
            vals = []
            for c in cols:
                vals.append(str(row[c]) if c == "target" else fmt(row[c], 4))
            lines.append("| " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "Read discipline:",
            "",
            "- `corpus_lexical_only` is a corpus-level control and may behave like a corpus-regime proxy.",
            "- `corpus_lexical_plus_field` exceeding `corpus_lexical_only` suggests field geometry carries signal beyond corpus lexical fingerprint.",
            "- `field_no_direct_seam_no_grid` is the strongest field-only anti-shortcut comparison.",
            "- `corpus_lexical_plus_field_no_direct_seam_no_grid` is the strongest lexical-bridge anti-shortcut comparison.",
            "- Path-level lexical controls are only valid when the reported path-id overlap is nontrivial.",
            "",
            "## Top permutation importances",
            "",
        ]
    )

    if perm.empty:
        lines.append("No permutation importances.")
    else:
        for target in [t.name for t in TARGET_SPECS]:
            for fs in [
                "field_only",
                "field_no_direct_seam_no_grid",
                "corpus_lexical_only",
                "corpus_lexical_plus_field",
                "corpus_lexical_plus_field_no_direct_seam_no_grid",
                "path_lexical_only",
                "path_lexical_plus_field_no_direct_seam_no_grid",
            ]:
                sub = perm[(perm["target"].eq(target)) & (perm["feature_set"].eq(fs))].sort_values("rank")
                if sub.empty:
                    continue
                lines.append(f"### {target} / {fs}")
                lines.append("")
                for _, row in sub.head(cfg.n_top_importances).iterrows():
                    lines.append(
                        f"- `{row['feature']}` [{row.get('feature_provenance', '')}]: "
                        f"{fmt(row['importance_mean'], 6)} ± {fmt(row['importance_std'], 6)}"
                    )
                lines.append("")

    lines.extend(["", "## Feature-manifest summary", ""])
    if feature_manifest.empty:
        lines.append("No feature manifest written.")
    else:
        summary = (
            feature_manifest
            .groupby(["target", "feature_set", "allowed", "provenance_or_exclusion"], as_index=False)
            .agg(n_features=("feature", "count"))
            .sort_values(["target", "feature_set", "allowed", "provenance_or_exclusion"])
        )
        lines.append("| target | feature_set | allowed | provenance_or_exclusion | n_features |")
        lines.append("| --- | --- | ---: | --- | ---: |")
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['target']} | {row['feature_set']} | {int(row['allowed'])} | "
                f"{row['provenance_or_exclusion']} | {int(row['n_features'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Lexical predictability does not falsify field geometry; it identifies a possible surface confound.",
            "- Field predictability after lexical controls is stronger evidence for geometric reduction.",
            "- Blinded field predictability after lexical controls is stronger evidence than unblinded field predictability.",
            "- Corpus-level lexical controls are not path-level lexical explanations.",
            "- Tokenizer-specific claims require a separate tokenizer-aligned analysis.",
            "- Corpus differences remain corpus/root comparisons, not universal model claims.",
            "- In the v3 smoke run, corpus-level lexical-only models remain far below blinded field-only models for all targets; path-level lexical controls are unavailable because no path-level text join was achieved.",
            "",
        ]
    )

    (cfg.outdir / "obs074_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_corpus_arg(raw: str, default_scale: str) -> CorpusSpec:
    """
    Format:
      LABEL=ROOT
      LABEL=ROOT::TEXT_JSON
      LABEL=ROOT::TEXT_JSON::SCALE
    """
    parts = raw.split("::")
    left = parts[0]
    if "=" not in left:
        raise ValueError(f"Invalid corpus spec; expected LABEL=ROOT: {raw}")

    label, root = left.split("=", 1)
    text_source = Path(parts[1]) if len(parts) >= 2 and parts[1] else None
    scale = parts[2] if len(parts) >= 3 and parts[2] else default_scale
    return CorpusSpec(label=label, root=Path(root), text_source=text_source, scale=scale)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS-074 lexical-field bridge v3.")

    p.add_argument(
        "--corpus",
        action="append",
        required=True,
        help=(
            "Corpus spec. Format LABEL=ROOT or LABEL=ROOT::TEXT_JSON or "
            "LABEL=ROOT::TEXT_JSON::SCALE. Repeat for multiple corpora."
        ),
    )
    p.add_argument("--scale", default="100000")
    p.add_argument("--outdir", default="outputs/obs074_lexical_field_bridge_v3")

    p.add_argument("--random-state", type=int, default=74)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--max-rows-per-corpus", type=int, default=0)

    p.add_argument("--rf-n-estimators", type=int, default=500)
    p.add_argument("--rf-max-depth", type=int, default=5)
    p.add_argument("--rf-min-samples-leaf", type=int, default=20)
    p.add_argument("--permutation-repeats", type=int, default=8)
    p.add_argument("--n-top-importances", type=int, default=10)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    corpora = [parse_corpus_arg(x, args.scale) for x in args.corpus]

    cfg = Config(
        corpora=corpora,
        outdir=Path(args.outdir),
        random_state=args.random_state,
        min_class_count=args.min_class_count,
        max_rows_per_corpus=args.max_rows_per_corpus or None,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        permutation_repeats=args.permutation_repeats,
        n_top_importances=args.n_top_importances,
    )

    ensure_dir(cfg.outdir)

    feature_tables = []
    manifests = []

    for corpus in cfg.corpora:
        print(f"==> OBS-074 v3 build corpus: {corpus.label}")
        ft, manifest, _path_lex, _corpus_lex = build_feature_table_for_corpus(corpus)

        if cfg.max_rows_per_corpus is not None and len(ft) > cfg.max_rows_per_corpus:
            ft = ft.sample(
                n=cfg.max_rows_per_corpus,
                random_state=cfg.random_state,
                replace=False,
            ).copy()

        feature_tables.append(ft)
        manifests.append(manifest)

    feature_table = pd.concat(feature_tables, ignore_index=True)
    feature_table.to_csv(cfg.outdir / "obs074_feature_table.csv", index=False)

    lexical_audit = build_lexical_join_audit(feature_table)
    lexical_fingerprint = corpus_fingerprint(feature_table)
    target_manifest = build_target_manifest(feature_table)

    lexical_audit.to_csv(cfg.outdir / "obs074_lexical_join_audit.csv", index=False)
    lexical_fingerprint.to_csv(cfg.outdir / "obs074_lexical_corpus_fingerprint.csv", index=False)
    target_manifest.to_csv(cfg.outdir / "obs074_target_manifest.csv", index=False)

    all_scores = []
    all_perm = []
    all_conf = []
    all_feature_manifest = []

    for target in TARGET_SPECS:
        for fs in FEATURE_SET_SPECS:
            print(f"==> OBS-074 v3 target={target.name} feature_set={fs.name}")

            scores, perm, conf, feat_manifest = evaluate_model(cfg, feature_table, target, fs)
            all_scores.append(scores)
            all_perm.append(perm)
            all_conf.append(conf)
            all_feature_manifest.append(feat_manifest)

    model_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    perm = pd.concat(all_perm, ignore_index=True) if all_perm else pd.DataFrame()
    confusion = pd.concat(all_conf, ignore_index=True) if all_conf else pd.DataFrame()
    feature_manifest = pd.concat(all_feature_manifest, ignore_index=True) if all_feature_manifest else pd.DataFrame()

    read_table = lexical_vs_field_read_table(model_scores)

    model_scores.to_csv(cfg.outdir / "obs074_model_scores.csv", index=False)
    model_scores.to_csv(cfg.outdir / "obs074_ablation_scores.csv", index=False)
    perm.to_csv(cfg.outdir / "obs074_feature_importance_permutation.csv", index=False)
    confusion.to_csv(cfg.outdir / "obs074_confusion_matrices.csv", index=False)
    feature_manifest.to_csv(cfg.outdir / "obs074_feature_manifest.csv", index=False)
    read_table.to_csv(cfg.outdir / "obs074_lexical_vs_field_read.csv", index=False)

    write_summary(
        cfg=cfg,
        manifests=manifests,
        feature_table=feature_table,
        lexical_audit=lexical_audit,
        lexical_fingerprint=lexical_fingerprint,
        model_scores=model_scores,
        read_table=read_table,
        perm=perm,
        feature_manifest=feature_manifest,
    )

    print(cfg.outdir / "obs074_feature_table.csv")
    print(cfg.outdir / "obs074_lexical_join_audit.csv")
    print(cfg.outdir / "obs074_lexical_corpus_fingerprint.csv")
    print(cfg.outdir / "obs074_target_manifest.csv")
    print(cfg.outdir / "obs074_model_scores.csv")
    print(cfg.outdir / "obs074_lexical_vs_field_read.csv")
    print(cfg.outdir / "obs074_feature_importance_permutation.csv")
    print(cfg.outdir / "obs074_confusion_matrices.csv")
    print(cfg.outdir / "obs074_feature_manifest.csv")
    print(cfg.outdir / "obs074_summary.md")


if __name__ == "__main__":
    main()
