#!/usr/bin/env python3
"""
OBS-075d — Cp3 path-level lexical control / audit.

Purpose
-------
Test whether Cp3→Cp2 directional asymmetry from OBS-075b/075c survives
path-level lexical controls where actual response text can be joined.

The script is deliberately audit-first:

1. Load cross-corpus path feature tables if available.
2. Load response text JSON files.
3. Attempt path-level lexical joins.
4. If the join succeeds, rerun cross-corpus transfer asymmetry with:
   - field_only
   - path_lexical_only
   - path_lexical_plus_field
   - corpus_lexical_only
   - corpus_lexical_plus_field
   and blinded variants.
5. If the join fails, emit explicit audit artifacts rather than silently
   claiming a lexical control.

Typical invocation
------------------
PYTHONPATH=src .venv/bin/python experiments/studies/obs075d_cp3_path_lexical_control.py \\
  --run obs075b=outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_logreg \\
  --text Cp3=observatory/corpora/Cp3.json \\
  --text Cp2=observatory/corpora/Cp2.json \\
  --text Cp=observatory/corpora/Cp.json \\
  --text C=observatory/corpora/C.json \\
  --feature-table Cp3=outputs/comparisons/obs074_Cp3_lexical_field_bridge_smoke_v2/obs074_feature_table.csv \\
  --feature-table Cp2=outputs/comparisons/obs074_Cp2_lexical_field_bridge_smoke_v2/obs074_feature_table.csv \\
  --feature-table Cp=outputs/comparisons/obs074_Cp_lexical_field_bridge_smoke_v2/obs074_feature_table.csv \\
  --feature-table C=outputs/comparisons/obs074_C_lexical_field_bridge_smoke_v2/obs074_feature_table.csv \\
  --outdir outputs/comparisons/obs075d_cp3_path_lexical_control

Notes
-----
- If no --feature-table is supplied, the script searches supplied --run
  directories for likely OBS-075b feature-table artifacts.
- If no path-level text join is possible, corpus-level lexical fingerprints
  can still be attached, but path-level lexical claims remain unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN_CANDIDATES = {
    "path_family": ["target_path_family", "path_family"],
    "coupling_class": ["target_coupling_class", "coupling_class"],
    "outcome_group": ["target_outcome_group", "outcome_group"],
    "coupled_outcome_group": [
        "target_coupled_outcome_group",
        "coupled_outcome_group",
    ],
    "recovery_channel_structural": [
        "target_recovery_channel_structural",
        "recovery_channel_structural",
    ],
    "recovery_channel": [
        "target_recovery_channel",
        "recovery_channel",
        "target_recovery_channel_structural",
        "recovery_channel_structural",
    ],
    "recovery_channel_boundedness_strict": [
        "target_recovery_channel_structural",
        "recovery_channel_structural",
        "target_recovery_channel",
        "recovery_channel",
    ],
}

DEFAULT_TARGETS = [
    "coupled_outcome_group",
    "recovery_channel_structural",
    "recovery_channel_boundedness_strict",
    "coupling_class",
    "outcome_group",
    "path_family",
]

PAIR_LABELS = ["Cp3", "Cp2", "Cp", "C"]


@dataclass(frozen=True)
class CorpusSpec:
    label: str
    feature_table: Path | None = None
    text_path: Path | None = None


def parse_key_path(spec: str, arg_name: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"{arg_name} must use LABEL=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise SystemExit(f"{arg_name} has empty label: {spec}")
    if not path:
        raise SystemExit(f"{arg_name} has empty path: {spec}")
    return label, Path(path)


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] failed to read CSV {path}: {exc}", file=sys.stderr)
        return None


def discover_feature_tables(run_dirs: dict[str, Path]) -> dict[str, Path]:
    """
    Search OBS-075b/OBS-075d-like run dirs for reusable feature tables.

    The script remains conservative: discovery is best-effort only.
    """
    candidates: dict[str, Path] = {}
    likely_names = [
        "obs075b_feature_table.csv",
        "obs075b_training_table.csv",
        "obs075b_joined_feature_table.csv",
        "obs075d_feature_table.csv",
        "feature_table.csv",
    ]

    for _, root in run_dirs.items():
        if not root.exists():
            continue

        for name in likely_names:
            p = root / name
            if p.exists():
                df = safe_read_csv(p)
                if df is None or "corpus" not in df.columns:
                    continue
                for label in sorted(df["corpus"].dropna().astype(str).unique()):
                    candidates.setdefault(label, p)

        for p in root.glob("**/*.csv"):
            if p.name in likely_names:
                continue
            if "feature" not in p.name.lower() and "table" not in p.name.lower():
                continue
            df = safe_read_csv(p)
            if df is None or "corpus" not in df.columns:
                continue
            for label in sorted(df["corpus"].dropna().astype(str).unique()):
                candidates.setdefault(label, p)

    return candidates


def flatten_json_records(obj: Any) -> list[dict[str, Any]]:
    """
    Extract text records from common corpus JSON shapes.

    Supports:
    - list[str]
    - list[dict]
    - {"C": [str, ...]}
    - {"records": [...]}
    - {"items": [...]}
    - {"responses": [...]}
    - nested dict/list structures
    """
    records: list[dict[str, Any]] = []

    def walk(x: Any, inherited_key: str | None = None) -> None:
        if isinstance(x, str):
            if x.strip():
                records.append(
                    {
                        "text": x,
                        "source_key": inherited_key or "",
                    }
                )
            return

        if isinstance(x, dict):
            if any(k in x for k in ("text", "response", "content", "completion", "message", "answer", "output")):
                records.append(x)

            for key, val in x.items():
                if isinstance(val, (list, dict, str)):
                    walk(val, inherited_key=key)
            return

        if isinstance(x, list):
            for item in x:
                walk(item, inherited_key=inherited_key)
            return

    walk(obj)
    return records


def pick_first_str(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def extract_text_from_record(record: dict[str, Any]) -> str:
    direct = pick_first_str(
        record,
        [
            "text",
            "response",
            "content",
            "completion",
            "message",
            "answer",
            "output",
        ],
    )
    if direct is not None:
        return direct

    for key in ("choices", "messages"):
        val = record.get(key)
        if isinstance(val, list):
            parts: list[str] = []
            for item in val:
                if isinstance(item, dict):
                    part = pick_first_str(item, ["text", "content", "message"])
                    if part:
                        parts.append(part)
            if parts:
                return "\n".join(parts)

    return ""


def extract_id_from_record(record: dict[str, Any]) -> str | None:
    for key in (
        "path_id",
        "id",
        "sample_id",
        "trajectory_id",
        "job_id",
        "prompt_id",
        "response_id",
    ):
        val = record.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    meta = record.get("metadata")
    if isinstance(meta, dict):
        for key in ("path_id", "id", "sample_id", "trajectory_id", "job_id"):
            val = meta.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()

    return None


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
SENT_RE = re.compile(r"[.!?]+")


def lexical_features_for_text(text: str, prefix: str) -> dict[str, float]:
    text = text or ""
    chars = len(text)
    words = TOKEN_RE.findall(text.lower())
    word_count = len(words)
    types = len(set(words))
    lines = text.splitlines()
    sentences = [s for s in SENT_RE.split(text) if s.strip()]

    counts = Counter(words)
    trigrams = Counter(zip(words, words[1:], words[2:])) if len(words) >= 3 else Counter()

    def density(vocab: set[str]) -> float:
        if word_count == 0:
            return 0.0
        return sum(counts[w] for w in vocab) / word_count

    modal_words = {
        "may",
        "might",
        "could",
        "would",
        "should",
        "must",
        "can",
        "likely",
        "possibly",
        "probably",
    }
    causal_words = {
        "because",
        "therefore",
        "thus",
        "hence",
        "since",
        "so",
        "causes",
        "implies",
        "means",
        "leads",
    }
    uncertainty_words = {
        "uncertain",
        "unclear",
        "unknown",
        "ambiguous",
        "perhaps",
        "maybe",
        "possibly",
        "provisional",
    }
    mechanistic_words = {
        "operator",
        "field",
        "geometry",
        "manifold",
        "seam",
        "trajectory",
        "coupling",
        "phase",
        "gradient",
        "channel",
        "control",
        "ablation",
    }

    sentence_lengths = [len(TOKEN_RE.findall(s)) for s in sentences]

    return {
        f"{prefix}_word_count": float(word_count),
        f"{prefix}_char_count": float(chars),
        f"{prefix}_type_token_ratio": float(types / word_count) if word_count else 0.0,
        f"{prefix}_newline_density": float(text.count("\n") / max(chars, 1)),
        f"{prefix}_colon_density": float(text.count(":") / max(chars, 1)),
        f"{prefix}_semicolon_density": float(text.count(";") / max(chars, 1)),
        f"{prefix}_comma_density": float(text.count(",") / max(chars, 1)),
        f"{prefix}_markdown_bold_density": float(text.count("**") / max(chars, 1)),
        f"{prefix}_bullet_density": float(
            sum(1 for line in lines if line.strip().startswith(("-", "*", "•"))) / max(len(lines), 1)
        ),
        f"{prefix}_modal_density": density(modal_words),
        f"{prefix}_causal_density": density(causal_words),
        f"{prefix}_uncertainty_density": density(uncertainty_words),
        f"{prefix}_mechanistic_density": density(mechanistic_words),
        f"{prefix}_top_trigram_share": float(max(trigrams.values()) / max(sum(trigrams.values()), 1))
        if trigrams
        else 0.0,
        f"{prefix}_sentence_count": float(len(sentences)),
        f"{prefix}_mean_sentence_words": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        f"{prefix}_max_sentence_words": float(np.max(sentence_lengths)) if sentence_lengths else 0.0,
        f"{prefix}_empty_response": float(word_count == 0),
        f"{prefix}_cutoff_like": float(
            "cutoff" in text.lower()
            or "i don't have access" in text.lower()
            or "cannot access" in text.lower()
        ),
    }


def aggregate_lexical_features(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(r.keys() for r in rows)))
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        base = key.replace(f"{prefix}_", f"{prefix}_corpus_")
        out[f"{base}_mean"] = float(np.mean(vals))
        out[f"{base}_std"] = float(np.std(vals))
        out[f"{base}_min"] = float(np.min(vals))
        out[f"{base}_max"] = float(np.max(vals))
        out[f"{base}_median"] = float(np.median(vals))
    return out


def load_text_features(label: str, path: Path | None) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    if path is None:
        return pd.DataFrame(), {}, {
            "corpus": label,
            "text_source": "",
            "text_load_status": "missing_text_source",
            "n_text_rows": 0,
            "n_text_rows_with_id": 0,
        }

    if not path.exists():
        return pd.DataFrame(), {}, {
            "corpus": label,
            "text_source": str(path),
            "text_load_status": "missing_file",
            "n_text_rows": 0,
            "n_text_rows_with_id": 0,
        }

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = flatten_json_records(raw)
    except Exception as exc:
        return pd.DataFrame(), {}, {
            "corpus": label,
            "text_source": str(path),
            "text_load_status": f"load_error:{type(exc).__name__}",
            "n_text_rows": 0,
            "n_text_rows_with_id": 0,
        }

    rows = []
    lex_rows = []
    for i, rec in enumerate(records):
        text = extract_text_from_record(rec)
        rid = extract_id_from_record(rec)
        feats = lexical_features_for_text(text, prefix="lex_path")
        lex_rows.append(feats)
        row = {"text_row_index": i, "path_id": rid, "corpus": label}
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    corpus_feats = aggregate_lexical_features(lex_rows, prefix="lex_path")

    return df, corpus_feats, {
        "corpus": label,
        "text_source": str(path),
        "text_load_status": "ok",
        "n_text_rows": int(len(records)),
        "n_text_rows_with_id": int(df["path_id"].notna().sum()) if not df.empty else 0,
    }


def normalize_path_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def attach_lexical_features(
    feature_df: pd.DataFrame,
    text_df: pd.DataFrame,
    corpus_feats: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = feature_df.copy()

    if "path_id" not in df.columns:
        df["path_id"] = np.nan

    df["path_id"] = normalize_path_id_series(df["path_id"])
    if not text_df.empty and "path_id" in text_df.columns:
        t = text_df.copy()
        t = t[t["path_id"].notna()].copy()
        t["path_id"] = normalize_path_id_series(t["path_id"])
        t = t.drop_duplicates("path_id", keep="first")

        before = len(df)
        merged = df.merge(
            t.drop(columns=["corpus"], errors="ignore"),
            how="left",
            on="path_id",
            suffixes=("", "_text"),
        )
        lex_cols = [c for c in merged.columns if c.startswith("lex_path_")]
        overlap = int(merged[lex_cols].notna().any(axis=1).sum()) if lex_cols else 0
        mode = "path_id" if overlap > 0 else "none"
        df = merged
    else:
        before = len(df)
        overlap = 0
        mode = "none"

    for key, val in corpus_feats.items():
        df[key] = val

    audit = {
        "lexical_path_join_mode": mode,
        "lexical_path_overlap_rows": overlap,
        "lexical_path_overlap_share": float(overlap / max(before, 1)),
        "lexical_path_rows": int(len(text_df)) if text_df is not None else 0,
        "n_feature_rows": int(before),
        "n_corpus_lexical_features": int(len(corpus_feats)),
    }
    return df, audit


def infer_target_column(df: pd.DataFrame, target: str) -> str | None:
    candidates = TARGET_COLUMN_CANDIDATES.get(target, [])
    for c in candidates:
        if c in df.columns:
            return c

    stripped = target
    for suffix in (
        "_no_direct_seam_no_grid",
        "_boundedness_strict",
        "_structural",
    ):
        stripped = stripped.replace(suffix, "")
    for c in TARGET_COLUMN_CANDIDATES.get(stripped, []):
        if c in df.columns:
            return c
    return None


def is_target_or_metadata(col: str) -> bool:
    if col in {
        "path_id",
        "corpus",
        "source_root",
        "scale",
        "text_row_index",
    }:
        return True
    if col.startswith("target_"):
        return True
    if col in {
        "path_family",
        "coupling_class",
        "outcome_group",
        "coupled_outcome_group",
        "recovery_channel",
        "recovery_channel_structural",
    }:
        return True
    if col.startswith("obs050_"):
        return True
    return False


def is_direct_seam_feature(col: str) -> bool:
    c = col.lower()
    return (
        "distance_to_seam" in c
        or "near_fraction" in c
        or "mid_fraction" in c
        or "far_fraction" in c
        or re.search(r"(^|_)seam($|_)", c) is not None
    )


def is_grid_location_feature(col: str) -> bool:
    c = col.lower()
    if "last_minus_first" in c:
        return False
    grid_tokens = [
        "node_id_x",
        "node_id_y",
        "mds1",
        "mds2",
        "_r_",
        "_alpha_",
    ]
    return any(tok in c for tok in grid_tokens)


def is_endpoint_velocity_feature(col: str) -> bool:
    c = col.lower()
    return (
        "endpoint" in c
        or "velocity" in c
        or "last_minus_first" in c
        or c.endswith("_first")
        or c.endswith("_last")
    )


def is_tortuosity_feature(col: str) -> bool:
    c = col.lower()
    return "tortuosity" in c or "angle_jump" in c or "sector_change" in c


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if is_target_or_metadata(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def select_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    cols = numeric_feature_columns(df)

    include_field = "field" in feature_set or feature_set.startswith("no_")
    include_path_lex = "path_lexical" in feature_set
    include_corpus_lex = "corpus_lexical" in feature_set

    selected = []
    for c in cols:
        is_path_lex = c.startswith("lex_path_")
        is_corpus_lex = c.startswith("lex_path_corpus_") or c.startswith("lex_corpus_")
        is_lex = is_path_lex or is_corpus_lex

        if is_lex:
            if is_path_lex and not include_path_lex:
                continue
            if is_corpus_lex and not include_corpus_lex:
                continue
            if feature_set in {"path_lexical_only", "corpus_lexical_only"}:
                selected.append(c)
                continue
        else:
            if not include_field and feature_set not in {"field_only"}:
                continue

        if "no_direct_seam" in feature_set and is_direct_seam_feature(c):
            continue
        if "no_grid" in feature_set and is_grid_location_feature(c):
            continue
        if "no_endpoint_velocity" in feature_set and is_endpoint_velocity_feature(c):
            continue
        if "no_tortuosity" in feature_set and is_tortuosity_feature(c):
            continue

        if feature_set == "path_lexical_only" and not is_path_lex:
            continue
        if feature_set == "corpus_lexical_only" and not is_corpus_lex:
            continue
        if feature_set == "field_only" and is_lex:
            continue

        selected.append(c)

    return sorted(set(selected))


def make_model(model_family: str, random_state: int, max_depth: int | None) -> Pipeline:
    if model_family == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            random_state=random_state,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    if model_family == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=max_depth,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )

    raise ValueError(f"unsupported model_family: {model_family}")


def pair_score(
    df: pd.DataFrame,
    source: str,
    dest: str,
    target_col: str,
    feature_cols: list[str],
    model_family: str,
    random_state: int,
    max_depth: int | None,
    min_class_count: int,
) -> dict[str, Any]:
    src = df[(df["corpus"] == source) & df[target_col].notna()].copy()
    dst = df[(df["corpus"] == dest) & df[target_col].notna()].copy()

    out = {
        "source": source,
        "dest": dest,
        "target_col": target_col,
        "n_source": int(len(src)),
        "n_dest": int(len(dst)),
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "status": "unavailable",
    }

    if not feature_cols:
        out["status"] = "no_features"
        return out

    if len(src) == 0 or len(dst) == 0:
        out["status"] = "missing_rows"
        return out

    shared_classes = sorted(
        set(src[target_col].astype(str).unique()).intersection(
            set(dst[target_col].astype(str).unique())
        )
    )
    if len(shared_classes) < 2:
        out["status"] = "insufficient_shared_classes"
        return out

    src = src[src[target_col].astype(str).isin(shared_classes)].copy()
    dst = dst[dst[target_col].astype(str).isin(shared_classes)].copy()

    src_counts = src[target_col].astype(str).value_counts()
    dst_counts = dst[target_col].astype(str).value_counts()
    if src_counts.min() < min_class_count or dst_counts.min() < min_class_count:
        out["status"] = "insufficient_class_count"
        out["classes"] = json.dumps(shared_classes)
        return out

    X_src = src[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_src = src[target_col].astype(str)
    X_dst = dst[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_dst = dst[target_col].astype(str)

    try:
        model = make_model(model_family, random_state, max_depth)
        model.fit(X_src, y_src)
        pred = model.predict(X_dst)
        out.update(
            {
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_dst, pred, adjusted=False)
                ),
                "macro_f1": float(f1_score(y_dst, pred, average="macro")),
                "weighted_f1": float(f1_score(y_dst, pred, average="weighted")),
                "status": "ok",
                "classes": json.dumps(shared_classes),
            }
        )
    except Exception as exc:
        out["status"] = f"fit_or_eval_error:{type(exc).__name__}"

    return out


def get_ba(pair_df: pd.DataFrame, src: str, dst: str) -> float:
    m = pair_df[(pair_df["source"] == src) & (pair_df["dest"] == dst)]
    if m.empty:
        return np.nan
    val = m.iloc[0]["balanced_accuracy"]
    return float(val) if pd.notna(val) else np.nan


def asymmetry_rows(pair_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["target", "target_col", "feature_set", "model_family", "max_depth"]
    for keys, g in pair_scores.groupby(group_cols, dropna=False):
        target, target_col, feature_set, model_family, max_depth = keys

        ba_cp2_to_cp3 = get_ba(g, "Cp2", "Cp3")
        ba_cp3_to_cp2 = get_ba(g, "Cp3", "Cp2")
        ba_cp_to_cp3 = get_ba(g, "Cp", "Cp3")
        ba_cp3_to_cp = get_ba(g, "Cp3", "Cp")
        ba_c_to_cp3 = get_ba(g, "C", "Cp3")
        ba_cp3_to_c = get_ba(g, "Cp3", "C")

        asym_cp3_cp2 = ba_cp3_to_cp2 - ba_cp2_to_cp3
        asym_cp3_cp = ba_cp3_to_cp - ba_cp_to_cp3
        asym_cp3_c = ba_cp3_to_c - ba_c_to_cp3

        spec_cp = asym_cp3_cp2 - asym_cp3_cp
        spec_c = asym_cp3_cp2 - asym_cp3_c

        statuses = {
            f"{r.source}->{r.dest}": r.status
            for r in g[["source", "dest", "status"]].itertuples(index=False)
        }

        rows.append(
            {
                "target": target,
                "target_col": target_col,
                "feature_set": feature_set,
                "model_family": model_family,
                "max_depth": max_depth,
                "ba_cp2_to_cp3": ba_cp2_to_cp3,
                "ba_cp3_to_cp2": ba_cp3_to_cp2,
                "ba_cp_to_cp3": ba_cp_to_cp3,
                "ba_cp3_to_cp": ba_cp3_to_cp,
                "ba_c_to_cp3": ba_c_to_cp3,
                "ba_cp3_to_c": ba_cp3_to_c,
                "asymmetry_cp3_minus_cp2": asym_cp3_cp2,
                "specificity_vs_cp": spec_cp,
                "specificity_vs_c": spec_c,
                "status_json": json.dumps(statuses, sort_keys=True),
                "specificity_status": specificity_status(
                    asym_cp3_cp2, spec_cp, spec_c
                ),
                "survival_read": survival_read(asym_cp3_cp2, spec_cp, spec_c),
            }
        )

    return pd.DataFrame(rows)


def finite_positive(x: float, eps: float = 0.02) -> bool:
    return x is not None and np.isfinite(x) and x > eps


def specificity_status(asym: float, spec_cp: float, spec_c: float) -> str:
    has_cp = np.isfinite(spec_cp)
    has_c = np.isfinite(spec_c)
    if has_cp and has_c:
        return "ok"
    if has_cp:
        return "ok_vs_cp_only"
    if has_c:
        return "ok_vs_c_only"
    if np.isfinite(asym):
        return "direction_only"
    return "unavailable_or_partial"


def survival_read(asym: float, spec_cp: float, spec_c: float) -> str:
    if not finite_positive(asym):
        return "collapsed_or_near_zero"

    cp = finite_positive(spec_cp)
    c = finite_positive(spec_c)

    if cp and c:
        return "survives_vs_cp_and_c"
    if cp:
        return "survives_vs_cp_only"
    if c:
        return "survives_vs_c_only"
    if np.isfinite(asym):
        return "directional_without_specificity"
    return "unavailable_or_partial"


def write_markdown_table(f, df: pd.DataFrame, max_rows: int | None = None) -> None:
    if df is None or df.empty:
        f.write("_No rows._\n\n")
        return
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows).copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
        else:
            out[c] = out[c].map(lambda x: "NA" if pd.isna(x) else str(x))
    f.write(out.to_markdown(index=False))
    f.write("\n\n")


def build_summary(
    outdir: Path,
    args: argparse.Namespace,
    text_manifest: pd.DataFrame,
    join_audit: pd.DataFrame,
    lexical_fingerprint: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    pair_scores: pd.DataFrame,
    asym_df: pd.DataFrame,
) -> None:
    path = outdir / "obs075d_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# OBS-075d — Cp3 path-level lexical control\n\n")

        f.write("## Scope\n\n")
        f.write(
            "OBS-075d audits whether Cp3 directional asymmetry can be tested "
            "against actual response-text lexical controls at path level. "
            "When path-level text identifiers are unavailable, this script "
            "records that limitation explicitly rather than treating corpus-level "
            "lexical fingerprints as path-level evidence.\n\n"
        )

        f.write("## Inputs\n\n")
        f.write(f"- model_family: `{args.model_family}`\n")
        f.write(f"- max_depth: `{args.max_depth}`\n")
        f.write(f"- min_class_count: `{args.min_class_count}`\n")
        f.write(f"- random_state: `{args.random_state}`\n\n")

        f.write("## Text manifest\n\n")
        write_markdown_table(f, text_manifest)

        f.write("## Lexical path join audit\n\n")
        write_markdown_table(f, join_audit)

        f.write("## Lexical corpus fingerprint\n\n")
        if lexical_fingerprint.empty:
            f.write("_No corpus lexical fingerprint available._\n\n")
        else:
            compact_cols = [
                c
                for c in lexical_fingerprint.columns
                if c == "corpus"
                or c.endswith("_word_count_mean")
                or c.endswith("_char_count_mean")
                or c.endswith("_type_token_ratio_mean")
                or c.endswith("_mechanistic_density_mean")
                or c.endswith("_modal_density_mean")
                or c.endswith("_uncertainty_density_mean")
                or c.endswith("_cutoff_like_mean")
            ]
            write_markdown_table(f, lexical_fingerprint[compact_cols])

        f.write("## Feature manifest\n\n")
        write_markdown_table(f, feature_manifest, max_rows=80)

        f.write("## Pair transfer scores\n\n")
        if pair_scores.empty:
            f.write(
                "_No pair transfer scores were produced. This usually means no "
                "usable feature tables were supplied or discovered._\n\n"
            )
        else:
            compact = pair_scores[
                [
                    "target",
                    "feature_set",
                    "source",
                    "dest",
                    "feature_count",
                    "n_source",
                    "n_dest",
                    "balanced_accuracy",
                    "macro_f1",
                    "status",
                ]
            ]
            write_markdown_table(f, compact, max_rows=120)

        f.write("## Cp3 directional asymmetry / specificity\n\n")
        if asym_df.empty:
            f.write("_No asymmetry rows were produced._\n\n")
        else:
            compact = asym_df[
                [
                    "target",
                    "feature_set",
                    "asymmetry_cp3_minus_cp2",
                    "specificity_vs_cp",
                    "specificity_vs_c",
                    "ba_cp2_to_cp3",
                    "ba_cp3_to_cp2",
                    "survival_read",
                    "specificity_status",
                ]
            ]
            write_markdown_table(f, compact, max_rows=120)

        f.write("## Interpretation guardrails\n\n")
        f.write(
            "- `path_join_none` means OBS-075d cannot test path-level lexical explanation from current artifacts.\n"
            "- Corpus-level lexical controls are not path-level lexical explanations.\n"
            "- `path_lexical_only` is meaningful only when path-level text overlap is nontrivial.\n"
            "- `field_plus_lexical ≈ field_only` suggests lexical features do not materially alter the field result.\n"
            "- A strong drop under `field_plus_lexical` suggests text-mediated or collinear substrate risk.\n"
            "- Survival under `no_direct_seam_no_grid_no_endpoint_velocity` is the strongest anti-shortcut read here.\n"
            "- Missing rows are not evidence of absence; they mark unavailable class support, feature support, or text provenance.\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OBS-075d Cp3 path-level lexical control / audit."
    )
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional baseline run dir. Format LABEL=DIR. Repeatable.",
    )
    ap.add_argument(
        "--feature-table",
        action="append",
        default=[],
        help="Feature table CSV. Format CORPUS=CSV. Repeatable.",
    )
    ap.add_argument(
        "--text",
        action="append",
        default=[],
        help="Response text JSON. Format CORPUS=JSON. Repeatable.",
    )
    ap.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target name. Repeatable. Defaults to canonical OBS-075 target set.",
    )
    ap.add_argument(
        "--feature-set",
        action="append",
        default=[],
        help="Feature set name. Repeatable. Defaults to OBS-075d canonical set.",
    )
    ap.add_argument(
        "--model-family",
        choices=["logreg", "rf"],
        default="logreg",
    )
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=17)
    ap.add_argument("--min-class-count", type=int, default=20)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/comparisons/obs075d_cp3_path_lexical_control"),
    )

    args = ap.parse_args()
    outdir = ensure_outdir(args.outdir)

    run_dirs = dict(parse_key_path(x, "--run") for x in args.run)
    feature_tables = dict(parse_key_path(x, "--feature-table") for x in args.feature_table)
    text_paths = dict(parse_key_path(x, "--text") for x in args.text)

    discovered = discover_feature_tables(run_dirs)
    for label, p in discovered.items():
        feature_tables.setdefault(label, p)

    labels = sorted(set(feature_tables) | set(text_paths) | set(PAIR_LABELS))

    text_manifest_rows = []
    join_audit_rows = []
    lexical_fingerprint_rows = []
    feature_manifest_rows = []
    joined_tables = []

    for label in labels:
        ft_path = feature_tables.get(label)
        text_path = text_paths.get(label)

        text_df, corpus_feats, text_manifest = load_text_features(label, text_path)
        text_manifest_rows.append(text_manifest)

        if corpus_feats:
            fp = {"corpus": label}
            fp.update(corpus_feats)
            lexical_fingerprint_rows.append(fp)

        if ft_path is None or not ft_path.exists():
            join_audit_rows.append(
                {
                    "corpus": label,
                    "feature_table": str(ft_path) if ft_path else "",
                    "feature_table_status": "missing",
                    "lexical_path_join_mode": "unavailable_no_feature_table",
                    "lexical_path_overlap_rows": 0,
                    "lexical_path_overlap_share": 0.0,
                    "lexical_path_rows": int(len(text_df)),
                    "n_feature_rows": 0,
                    "n_corpus_lexical_features": int(len(corpus_feats)),
                }
            )
            continue

        base = safe_read_csv(ft_path)
        if base is None or base.empty:
            join_audit_rows.append(
                {
                    "corpus": label,
                    "feature_table": str(ft_path),
                    "feature_table_status": "load_failed_or_empty",
                    "lexical_path_join_mode": "unavailable_no_feature_table",
                    "lexical_path_overlap_rows": 0,
                    "lexical_path_overlap_share": 0.0,
                    "lexical_path_rows": int(len(text_df)),
                    "n_feature_rows": 0,
                    "n_corpus_lexical_features": int(len(corpus_feats)),
                }
            )
            continue

        if "corpus" not in base.columns:
            base["corpus"] = label

        if len(base["corpus"].dropna().astype(str).unique()) > 1:
            sub = base[base["corpus"].astype(str) == label].copy()
        else:
            sub = base.copy()
            sub["corpus"] = label

        joined, audit = attach_lexical_features(sub, text_df, corpus_feats)
        audit.update(
            {
                "corpus": label,
                "feature_table": str(ft_path),
                "feature_table_status": "ok",
            }
        )
        join_audit_rows.append(audit)
        joined_tables.append(joined)

    text_manifest = pd.DataFrame(text_manifest_rows)
    join_audit = pd.DataFrame(join_audit_rows)
    lexical_fingerprint = pd.DataFrame(lexical_fingerprint_rows)

    text_manifest.to_csv(outdir / "obs075d_text_manifest.csv", index=False)
    join_audit.to_csv(outdir / "obs075d_join_audit.csv", index=False)
    lexical_fingerprint.to_csv(
        outdir / "obs075d_lexical_corpus_fingerprint.csv", index=False
    )

    if joined_tables:
        all_df = pd.concat(joined_tables, ignore_index=True, sort=False)
    else:
        all_df = pd.DataFrame()

    if not all_df.empty:
        all_df.to_csv(outdir / "obs075d_feature_table.csv", index=False)

    targets = args.target or DEFAULT_TARGETS
    feature_sets = args.feature_set or [
        "field_only",
        "field_no_direct_seam_no_grid",
        "field_no_direct_seam_no_grid_no_endpoint_velocity",
        "path_lexical_only",
        "path_lexical_plus_field",
        "path_lexical_plus_field_no_direct_seam_no_grid",
        "path_lexical_plus_field_no_direct_seam_no_grid_no_endpoint_velocity",
        "corpus_lexical_only",
        "corpus_lexical_plus_field",
        "corpus_lexical_plus_field_no_direct_seam_no_grid",
        "corpus_lexical_plus_field_no_direct_seam_no_grid_no_endpoint_velocity",
    ]

    pair_rows = []

    if not all_df.empty and "corpus" in all_df.columns:
        available_labels = set(all_df["corpus"].dropna().astype(str).unique())

        for target in targets:
            target_col = infer_target_column(all_df, target)
            if target_col is None:
                continue

            for feature_set in feature_sets:
                feature_cols = select_features(all_df, feature_set)
                feature_manifest_rows.append(
                    {
                        "target": target,
                        "target_col": target_col,
                        "feature_set": feature_set,
                        "feature_count": len(feature_cols),
                        "has_path_lexical_features": int(
                            any(c.startswith("lex_path_") and not c.startswith("lex_path_corpus_") for c in feature_cols)
                        ),
                        "has_corpus_lexical_features": int(
                            any(c.startswith("lex_path_corpus_") or c.startswith("lex_corpus_") for c in feature_cols)
                        ),
                        "has_field_features": int(
                            any(not c.startswith("lex_") for c in feature_cols)
                        ),
                    }
                )

                for src in PAIR_LABELS:
                    for dst in PAIR_LABELS:
                        if src == dst:
                            continue
                        if src not in available_labels or dst not in available_labels:
                            continue

                        row = pair_score(
                            all_df,
                            source=src,
                            dest=dst,
                            target_col=target_col,
                            feature_cols=feature_cols,
                            model_family=args.model_family,
                            random_state=args.random_state,
                            max_depth=args.max_depth,
                            min_class_count=args.min_class_count,
                        )
                        row.update(
                            {
                                "target": target,
                                "feature_set": feature_set,
                                "feature_count": len(feature_cols),
                                "model_family": args.model_family,
                                "max_depth": args.max_depth,
                            }
                        )
                        pair_rows.append(row)

    feature_manifest = pd.DataFrame(feature_manifest_rows)
    pair_scores = pd.DataFrame(pair_rows)

    feature_manifest.to_csv(outdir / "obs075d_feature_manifest.csv", index=False)
    pair_scores.to_csv(outdir / "obs075d_pair_transfer_scores.csv", index=False)

    if not pair_scores.empty:
        asym_df = asymmetry_rows(pair_scores)
    else:
        asym_df = pd.DataFrame()

    asym_df.to_csv(outdir / "obs075d_asymmetry_specificity.csv", index=False)

    build_summary(
        outdir=outdir,
        args=args,
        text_manifest=text_manifest,
        join_audit=join_audit,
        lexical_fingerprint=lexical_fingerprint,
        feature_manifest=feature_manifest,
        pair_scores=pair_scores,
        asym_df=asym_df,
    )

    print(outdir / "obs075d_text_manifest.csv")
    print(outdir / "obs075d_join_audit.csv")
    print(outdir / "obs075d_lexical_corpus_fingerprint.csv")
    if not all_df.empty:
        print(outdir / "obs075d_feature_table.csv")
    print(outdir / "obs075d_feature_manifest.csv")
    print(outdir / "obs075d_pair_transfer_scores.csv")
    print(outdir / "obs075d_asymmetry_specificity.csv")
    print(outdir / "obs075d_summary.md")


if __name__ == "__main__":
    main()
