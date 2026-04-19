#!/usr/bin/env python3
"""
obs0xx_route_family_recoverability.py

First-pass recoverability study for canonical seam-route families.

Purpose
-------
Test how recoverable route-family identity is from progressively richer feature
sets:

    A. pointwise local state
    B. distributed local neighborhood
    C. short route context
    D. broader route context

This is designed as a clean, testable bridge to distributed recoverability
questions:
- is family identity mostly pointwise?
- does local neighborhood support matter?
- does broader path context dominate?

The script is intentionally conservative:
- prefers simple, interpretable models
- uses multinomial logistic regression by default
- reports macro-F1, accuracy, and per-family recall
- supports any CSV with a family label and numeric features

Expected input
--------------
A CSV with one row per labeled path / transition / route instance.

Required:
    - family_label

Recommended:
    - multiple numeric columns spanning:
        * local state observables
        * local neighborhood summaries
        * short-context summaries
        * broader-context summaries

Feature grouping
----------------
You can specify feature groups explicitly with CLI flags, or rely on simple
prefix-based auto-grouping.

Auto-group prefixes:
    local_*
    nbr_*
    short_*
    broad_*

Feature-set ladder:
    A = local
    B = local + neighborhood
    C = local + neighborhood + short
    D = local + neighborhood + short + broad

Outputs
-------
Directory:
    outputs/obs0xx_route_family_recoverability/

Files:
    recoverability_summary.csv
    recoverability_per_family.csv
    recoverability_feature_sets.json
    recoverability_summary.txt
    recoverability_macro_f1.png
    recoverability_accuracy.png

Usage
-----
# Auto-group by prefixes
python experiments/studies/obs0xx_route_family_recoverability.py \
    --input outputs/canonical/trajectory_family_rollup.csv

# Explicit grouping
python experiments/studies/obs0xx_route_family_recoverability.py \
    --input my_family_dataset.csv \
    --label-col family_label \
    --local-cols distance_to_seam signed_phase lazarus_score \
    --neighborhood-cols nbr_mean_distance_to_seam nbr_mean_mismatch \
    --short-cols prev_transition_type short_prefix_turn short_prefix_gateway \
    --broad-cols temporal_depth forgetting_share compression_state_entropy

Notes
-----
- Categorical non-label columns are ignored unless one-hot encoded already.
- Rows with missing label are dropped.
- Numeric missing values are median-imputed within each fold.
- By default the script uses repeated stratified CV for stability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_OUTDIR = Path("outputs/obs0xx_route_family_recoverability")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route-family recoverability ladder study.")
    p.add_argument("--input", type=Path, required=True, help="CSV with family labels and features.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    p.add_argument("--label-col", type=str, default="family_label", help="Target label column.")

    p.add_argument("--local-cols", nargs="*", default=None, help="Explicit local feature columns.")
    p.add_argument("--neighborhood-cols", nargs="*", default=None, help="Explicit neighborhood feature columns.")
    p.add_argument("--short-cols", nargs="*", default=None, help="Explicit short-context feature columns.")
    p.add_argument("--broad-cols", nargs="*", default=None, help="Explicit broad-context feature columns.")

    p.add_argument("--local-prefix", type=str, default="local_", help="Auto-group prefix for local features.")
    p.add_argument("--neighborhood-prefix", type=str, default="nbr_", help="Auto-group prefix for neighborhood features.")
    p.add_argument("--short-prefix", type=str, default="short_", help="Auto-group prefix for short-context features.")
    p.add_argument("--broad-prefix", type=str, default="broad_", help="Auto-group prefix for broad-context features.")

    p.add_argument("--n-splits", type=int, default=5, help="Stratified CV folds.")
    p.add_argument("--n-repeats", type=int, default=10, help="Repeated CV repeats.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--min-class-count",
        type=int,
        default=3,
        help="Drop classes with fewer than this many rows before CV.",
    )
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return pd.read_csv(path)


def filter_numeric_existing(df: pd.DataFrame, cols: Iterable[str] | None) -> list[str]:
    if cols is None:
        return []
    out: list[str] = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def infer_group_columns(
    df: pd.DataFrame,
    local_cols: list[str] | None,
    neighborhood_cols: list[str] | None,
    short_cols: list[str] | None,
    broad_cols: list[str] | None,
    local_prefix: str,
    neighborhood_prefix: str,
    short_prefix: str,
    broad_prefix: str,
    label_col: str,
) -> dict[str, list[str]]:
    if any(x is not None for x in [local_cols, neighborhood_cols, short_cols, broad_cols]):
        groups = {
            "A_local": filter_numeric_existing(df, local_cols),
            "B_neighborhood_add": filter_numeric_existing(df, neighborhood_cols),
            "C_short_add": filter_numeric_existing(df, short_cols),
            "D_broad_add": filter_numeric_existing(df, broad_cols),
        }
    else:
        numeric_cols = [
            c
            for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        groups = {
            "A_local": [c for c in numeric_cols if c.startswith(local_prefix)],
            "B_neighborhood_add": [c for c in numeric_cols if c.startswith(neighborhood_prefix)],
            "C_short_add": [c for c in numeric_cols if c.startswith(short_prefix)],
            "D_broad_add": [c for c in numeric_cols if c.startswith(broad_prefix)],
        }

    return groups


def build_feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    a = groups["A_local"]
    b = a + groups["B_neighborhood_add"]
    c = b + groups["C_short_add"]
    d = c + groups["D_broad_add"]

    # preserve order, deduplicate
    def dedupe(xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "A_pointwise_local": dedupe(a),
        "B_local_plus_neighborhood": dedupe(b),
        "C_plus_short_context": dedupe(c),
        "D_plus_broad_context": dedupe(d),
    }


def prepare_dataset(df: pd.DataFrame, label_col: str, min_class_count: int) -> tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    work = df.copy()
    work = work.dropna(subset=[label_col]).reset_index(drop=True)
    y = work[label_col].astype(str)

    vc = y.value_counts()
    keep = vc[vc >= min_class_count].index.tolist()
    work = work[y.isin(keep)].reset_index(drop=True)
    y = work[label_col].astype(str)

    if y.nunique() < 2:
        raise ValueError("Need at least 2 label classes after filtering.")
    return work, y


def evaluate_feature_set(
    df: pd.DataFrame,
    y_raw: pd.Series,
    feature_cols: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    if len(feature_cols) == 0:
        raise ValueError("Feature set is empty.")

    X = df[feature_cols].copy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw.astype(str))
    class_name_map = {str(i): name for i, name in enumerate(le.classes_)}

    pipe = Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_cols,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                                solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    accs: list[float] = []
    macro_f1s: list[float] = []
    weighted_f1s: list[float] = []
    reports: list[dict] = []

    for train_idx, test_idx in cv.split(X, y_enc):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_enc[train_idx]
        y_test = y_enc[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        accs.append(float(accuracy_score(y_test, y_pred)))
        macro_f1s.append(float(f1_score(y_test, y_pred, average="macro")))
        weighted_f1s.append(float(f1_score(y_test, y_pred, average="weighted")))

        rep = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        reports.append(rep)

    summary = {
        "n_features": len(feature_cols),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
    }

    # aggregate per-class recall / precision / f1
    per_family_rows = []
    metric_names = ["precision", "recall", "f1-score", "support"]

    report_keys = set()
    for r in reports:
        report_keys.update(r.keys())

    family_keys = [k for k in report_keys if k not in {"accuracy", "macro avg", "weighted avg"}]

    def family_sort_key(k: str):
        try:
            return int(k)
        except ValueError:
            return k

    for fam in sorted(family_keys, key=family_sort_key):
        fam_name = class_name_map.get(str(fam), str(fam))
        row = {"family_label": fam_name}
        for m in metric_names:
            vals = [float(r.get(fam, {}).get(m, np.nan)) for r in reports]
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_std"] = float(np.nanstd(vals))
        per_family_rows.append(row)

    per_family_df = pd.DataFrame(per_family_rows)
    return summary, per_family_df


def plot_metric(summary_df: pd.DataFrame, metric: str, ylabel: str, outpath: Path) -> None:
    plot_df = summary_df.copy()
    x = np.arange(len(plot_df))
    vals = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, vals, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["feature_set"], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_summary_text(
    summary_df: pd.DataFrame,
    per_family_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    outpath: Path,
) -> None:
    best = summary_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
    lines = []
    lines.append("Route-family recoverability summary")
    lines.append("")
    lines.append(f"Feature sets tested: {len(summary_df)}")
    lines.append("")
    lines.append("Feature-set sizes:")
    for k, cols in feature_sets.items():
        lines.append(f"- {k}: {len(cols)} features")
    lines.append("")
    lines.append("Best feature set by macro-F1:")
    lines.append(f"- feature_set: {best['feature_set']}")
    lines.append(f"- n_features: {int(best['n_features'])}")
    lines.append(f"- macro_f1_mean: {best['macro_f1_mean']:.6f}")
    lines.append(f"- accuracy_mean: {best['accuracy_mean']:.6f}")
    lines.append("")
    lines.append("Per-family recall by feature set:")
    for _, row in per_family_df.iterrows():
        lines.append(
            f"- {row['feature_set']} / {row['family_label']}: recall_mean={row['recall_mean']:.6f}"
        )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_raw = read_csv(args.input)
    df, y = prepare_dataset(df_raw, args.label_col, args.min_class_count)

    groups = infer_group_columns(
        df=df,
        local_cols=args.local_cols,
        neighborhood_cols=args.neighborhood_cols,
        short_cols=args.short_cols,
        broad_cols=args.broad_cols,
        local_prefix=args.local_prefix,
        neighborhood_prefix=args.neighborhood_prefix,
        short_prefix=args.short_prefix,
        broad_prefix=args.broad_prefix,
        label_col=args.label_col,
    )
    feature_sets = build_feature_sets(groups)

    # Keep only non-empty feature sets, preserving ladder order
    feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}
    if len(feature_sets) == 0:
        raise ValueError("No usable feature sets found. Provide explicit columns or matching prefixes.")

    summary_rows = []
    per_family_frames = []

    for name, cols in feature_sets.items():
        summary, per_family = evaluate_feature_set(
            df=df,
            y_raw=y,
            feature_cols=cols,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            seed=args.seed,
        )
        summary["feature_set"] = name
        summary_rows.append(summary)

        per_family["feature_set"] = name
        per_family_frames.append(per_family)

    summary_df = pd.DataFrame(summary_rows)
    order = list(feature_sets.keys())
    summary_df["feature_set"] = pd.Categorical(summary_df["feature_set"], categories=order, ordered=True)
    summary_df = summary_df.sort_values("feature_set").reset_index(drop=True)

    per_family_df = pd.concat(per_family_frames, ignore_index=True)
    per_family_df["feature_set"] = pd.Categorical(per_family_df["feature_set"], categories=order, ordered=True)
    per_family_df = per_family_df.sort_values(["feature_set", "family_label"]).reset_index(drop=True)

    summary_path = args.outdir / "recoverability_summary.csv"
    per_family_path = args.outdir / "recoverability_per_family.csv"
    feature_sets_path = args.outdir / "recoverability_feature_sets.json"
    text_path = args.outdir / "recoverability_summary.txt"
    macro_fig = args.outdir / "recoverability_macro_f1.png"
    acc_fig = args.outdir / "recoverability_accuracy.png"

    summary_df.to_csv(summary_path, index=False)
    per_family_df.to_csv(per_family_path, index=False)
    feature_sets_path.write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    write_summary_text(summary_df, per_family_df, feature_sets, text_path)

    plot_metric(summary_df, "macro_f1_mean", "macro F1", macro_fig)
    plot_metric(summary_df, "accuracy_mean", "accuracy", acc_fig)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {per_family_path}")
    print(f"Wrote: {feature_sets_path}")
    print(f"Wrote: {text_path}")
    print(f"Wrote: {macro_fig}")
    print(f"Wrote: {acc_fig}")


if __name__ == "__main__":
    main()
