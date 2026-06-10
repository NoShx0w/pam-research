#!/usr/bin/env python3
"""
obs076d_scale_conditioned_factor_separability.py

OBS-076d — Scale-conditioned structural-factor separability

v2 patch
--------
v2 adds dynamic-only feature-family support.

v1 mixed dynamic OBS-076a columns and static/context OBS-076b columns in some
families, for example:

    phase:
      dyn__signed_phase + signed_phase

    response_lazarus:
      dyn__response_strength + response_strength

This was acceptable diagnostically, but weaker for provenance. v2 adds:

    --dynamic-only-families

When enabled, semantic families only use injected dynamic columns where possible:

    phase:
      dyn__signed_phase, dyn__distance_to_seam

    response_lazarus:
      dyn__response_strength, dyn__frobenius_T,
      dyn__trace_T, dyn__lazarus_score

    coupling:
      dyn__signed_coupling, dyn__cosine_alignment

It also adds explicit families:

    phase_dynamic
    response_lazarus_dynamic
    coupling_dynamic
    tensor_dynamic

Purpose
-------
OBS-076d consumes OBS-076a/076b/076c scale-space artifacts for two or more
cases and asks:

    At each diffusion scale, can simple classifiers distinguish cases using
    specific structural-factor feature families?

This is designed as the first classifier-facing bridge after OBS-076a/b/c.

For the current Cp2/Cp3 branch, the primary question is:

    Are Cp2 and Cp3 scale-factor arrangements linearly visible?
    At which scales?
    Through which feature families?

Scope discipline
----------------
This script does NOT test Cp3→Cp2 path-label transfer asymmetry directly.

It tests scale-conditioned corpus/case separability at node level.

It should be interpreted as a diagnostic bridge toward later path-level
or target-label transfer studies.

Inputs
------
Required:
  --case NAME=obs076b_node_geometry_by_scale.csv

Optional but recommended:
  --bundle NAME=obs076a_diffusion_bundle.npz
  --objects NAME=obs076c_object_membership_by_scale.csv

The OBS-076a bundle injects diffused observables as dyn__<observable>.
The OBS-076c memberships inject object-membership flags as obj__<object>.

Outputs
-------
outdir/
  obs076d_input_manifest.csv
  obs076d_feature_manifest.csv
  obs076d_scale_feature_table.csv
  obs076d_separability_scores.csv
  obs076d_feature_coefficients.csv
  obs076d_report.md

Guardrail
---------
High separability means the cases occupy different scale-conditioned
observable-space/factor supports. It does not identify causal mechanisms and
does not establish path-level transfer asymmetry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DYN_PREFIX = "dyn__"
OBJ_PREFIX = "obj__"


@dataclass(frozen=True)
class CaseInput:
    name: str
    node_geometry: Path


@dataclass(frozen=True)
class Config:
    cases: list[CaseInput]
    bundles: dict[str, Path]
    objects: dict[str, Path]
    outdir: Path
    id_col: str
    model_family: str
    cv_splits: int
    random_state: int
    min_class_count: int
    top_coefficients: int
    dynamic_only_families: bool


def parse_name_path(raw: str, arg_name: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"{arg_name} entries must be NAME=PATH; got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"{arg_name} entry has empty NAME: {raw!r}")
    if not path:
        raise ValueError(f"{arg_name} entry has empty PATH: {raw!r}")
    return name, Path(path)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="OBS-076d scale-conditioned structural-factor separability."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case input as NAME=obs076b_node_geometry_by_scale.csv. Provide at least two.",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Optional OBS-076a bundle as NAME=obs076a_diffusion_bundle.npz.",
    )
    parser.add_argument(
        "--objects",
        action="append",
        default=[],
        help="Optional OBS-076c object membership table as NAME=obs076c_object_membership_by_scale.csv.",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--id-col", default="id")
    parser.add_argument(
        "--model-family",
        choices=["logreg", "rf"],
        default="logreg",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--top-coefficients", type=int, default=25)
    parser.add_argument(
        "--dynamic-only-families",
        action="store_true",
        help=(
            "Use only dyn__ columns for semantic families where possible. "
            "This prevents static/context columns from entering phase, coupling, "
            "and response_lazarus families."
        ),
    )

    args = parser.parse_args()

    cases = [CaseInput(*parse_name_path(x, "--case")) for x in args.case]
    if len(cases) < 2:
        raise ValueError("Provide at least two --case entries")

    case_names = [c.name for c in cases]
    if len(case_names) != len(set(case_names)):
        raise ValueError(f"Duplicate case names: {case_names}")

    bundles = dict(parse_name_path(x, "--bundle") for x in args.bundle)
    objects = dict(parse_name_path(x, "--objects") for x in args.objects)

    unknown_bundle = sorted(set(bundles) - set(case_names))
    unknown_objects = sorted(set(objects) - set(case_names))
    if unknown_bundle:
        raise ValueError(f"--bundle contains names not present in --case: {unknown_bundle}")
    if unknown_objects:
        raise ValueError(f"--objects contains names not present in --case: {unknown_objects}")

    if args.cv_splits < 2:
        raise ValueError("--cv-splits must be >= 2")

    return Config(
        cases=cases,
        bundles=bundles,
        objects=objects,
        outdir=args.outdir,
        id_col=args.id_col,
        model_family=args.model_family,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        min_class_count=args.min_class_count,
        top_coefficients=args.top_coefficients,
        dynamic_only_families=bool(args.dynamic_only_families),
    )


def as_str_array(x: np.ndarray) -> np.ndarray:
    return np.array([str(v) for v in x.tolist()], dtype=str)


def load_node_geometry(case: CaseInput, id_col: str) -> pd.DataFrame:
    if not case.node_geometry.exists():
        raise FileNotFoundError(case.node_geometry)

    df = pd.read_csv(case.node_geometry)
    required = [id_col, "scale_index", "t"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{case.name}: node geometry missing required columns {missing}")

    df[id_col] = df[id_col].astype(str)
    df["scale_index"] = pd.to_numeric(df["scale_index"], errors="raise").astype(int)
    df["t"] = pd.to_numeric(df["t"], errors="raise").astype(float)
    df["case"] = case.name

    if df.duplicated(["case", id_col, "scale_index"]).any():
        dup = df[df.duplicated(["case", id_col, "scale_index"])][
            ["case", id_col, "scale_index"]
        ].head(10)
        raise ValueError(f"{case.name}: duplicate node/scale rows:\n{dup}")

    return df


def load_obs076a_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    required = ["ids", "observable_cols", "X_t", "t"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"OBS-076a bundle missing required keys: {missing}")

    ids = as_str_array(data["ids"])
    observable_cols = as_str_array(data["observable_cols"])
    x_t = np.asarray(data["X_t"], dtype=float)
    ts = np.asarray(data["t"], dtype=float)

    if x_t.ndim != 3:
        raise ValueError(f"X_t must be scales × nodes × observables; got {x_t.shape}")
    if x_t.shape[1] != len(ids):
        raise ValueError("X_t node dimension does not match ids")
    if x_t.shape[2] != len(observable_cols):
        raise ValueError("X_t observable dimension does not match observable_cols")
    if x_t.shape[0] != len(ts):
        raise ValueError("X_t scale dimension does not match t")

    return {"ids": ids, "observable_cols": observable_cols, "X_t": x_t, "t": ts}


def inject_bundle_features(
    df: pd.DataFrame,
    case_name: str,
    bundle_path: Path | None,
    id_col: str,
) -> tuple[pd.DataFrame, list[str], str]:
    if bundle_path is None:
        return df, [], "not_provided"

    bundle = load_obs076a_bundle(bundle_path)
    ids = bundle["ids"]
    obs_cols = bundle["observable_cols"]
    x_t = bundle["X_t"]
    ts = bundle["t"]

    scale_values = (
        df[["scale_index", "t"]]
        .drop_duplicates()
        .sort_values(["scale_index", "t"])
        .reset_index(drop=True)
    )

    if len(scale_values) != len(ts):
        raise ValueError(
            f"{case_name}: scale count {len(scale_values)} does not match bundle scale count {len(ts)}"
        )

    rows = []
    for sidx in range(len(ts)):
        block = pd.DataFrame({id_col: ids})
        block["scale_index"] = sidx
        block["bundle_t"] = float(ts[sidx])
        for j, col in enumerate(obs_cols):
            block[f"{DYN_PREFIX}{col}"] = x_t[sidx, :, j]
        rows.append(block)

    dyn = pd.concat(rows, ignore_index=True)
    dyn[id_col] = dyn[id_col].astype(str)
    dyn_cols = [c for c in dyn.columns if c.startswith(DYN_PREFIX)]

    merged = df.merge(dyn, on=[id_col, "scale_index"], how="left", validate="one_to_one")
    return merged, dyn_cols, "ok"


def load_object_memberships(
    path: Path,
    case_name: str,
    id_col: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    obj = pd.read_csv(path)
    required = [id_col, "scale_index", "object"]
    missing = [c for c in required if c not in obj.columns]
    if missing:
        raise ValueError(f"{case_name}: object membership missing columns {missing}")

    obj[id_col] = obj[id_col].astype(str)
    obj["scale_index"] = pd.to_numeric(obj["scale_index"], errors="raise").astype(int)
    obj["object"] = obj["object"].astype(str)

    obj["value"] = 1
    wide = (
        obj[[id_col, "scale_index", "object", "value"]]
        .drop_duplicates([id_col, "scale_index", "object"])
        .pivot_table(
            index=[id_col, "scale_index"],
            columns="object",
            values="value",
            fill_value=0,
            aggfunc="max",
        )
        .reset_index()
    )

    wide.columns = [
        c if c in {id_col, "scale_index"} else f"{OBJ_PREFIX}{c}"
        for c in wide.columns
    ]
    return wide


def inject_object_features(
    df: pd.DataFrame,
    case_name: str,
    objects_path: Path | None,
    id_col: str,
) -> tuple[pd.DataFrame, list[str], str]:
    if objects_path is None:
        return df, [], "not_provided"

    wide = load_object_memberships(objects_path, case_name, id_col)
    obj_cols = [c for c in wide.columns if c.startswith(OBJ_PREFIX)]

    merged = df.merge(
        wide,
        on=[id_col, "scale_index"],
        how="left",
        validate="one_to_one",
    )

    for c in obj_cols:
        merged[c] = merged[c].fillna(0).astype(int)

    return merged, obj_cols, "ok"


def load_all_cases(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifest_rows = []

    for case in cfg.cases:
        df = load_node_geometry(case, cfg.id_col)

        bundle_path = cfg.bundles.get(case.name)
        df, dyn_cols, dyn_status = inject_bundle_features(
            df=df,
            case_name=case.name,
            bundle_path=bundle_path,
            id_col=cfg.id_col,
        )

        objects_path = cfg.objects.get(case.name)
        df, obj_cols, obj_status = inject_object_features(
            df=df,
            case_name=case.name,
            objects_path=objects_path,
            id_col=cfg.id_col,
        )

        frames.append(df)

        manifest_rows.extend(
            [
                {
                    "case": case.name,
                    "artifact": "obs076b_node_geometry",
                    "path": str(case.node_geometry),
                    "status": "ok",
                    "details": "",
                },
                {
                    "case": case.name,
                    "artifact": "obs076a_bundle",
                    "path": str(bundle_path) if bundle_path else "",
                    "status": dyn_status,
                    "details": f"dynamic_columns={len(dyn_cols)}",
                },
                {
                    "case": case.name,
                    "artifact": "obs076c_objects",
                    "path": str(objects_path) if objects_path else "",
                    "status": obj_status,
                    "details": f"object_columns={len(obj_cols)}",
                },
            ]
        )

    all_df = pd.concat(frames, ignore_index=True, sort=False)

    for c in [c for c in all_df.columns if c.startswith(OBJ_PREFIX)]:
        all_df[c] = all_df[c].fillna(0).astype(int)

    manifest_df = pd.DataFrame(manifest_rows)
    return all_df, manifest_df


def feature_source_kind(col: str) -> str:
    if col.startswith(DYN_PREFIX):
        return "dynamic"
    if col.startswith(OBJ_PREFIX):
        return "object_membership"
    if col in {
        "energy",
        "density_score",
        "mean_knn_observable_distance",
        "seam_proxy_score",
        "is_seam_proxy",
        "phase_contrast",
        "geom_x",
        "geom_y",
    }:
        return "dynamic_proxy"
    return "static_or_context"


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "case",
        "case_label",
        "case_code",
        "scale_index",
        "t",
        "bundle_t",
    }

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c == "id" or c.endswith("_id"):
            continue
        if c in {"object", "source_column", "source_kind", "direction"}:
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        cols.append(c)

    return sorted(set(cols))


def family_columns(df: pd.DataFrame, dynamic_only_families: bool) -> dict[str, list[str]]:
    cols = numeric_feature_columns(df)

    def present(names: Iterable[str]) -> list[str]:
        return [c for c in names if c in cols]

    dyn_cols = [c for c in cols if c.startswith(DYN_PREFIX)]
    obj_cols = [c for c in cols if c.startswith(OBJ_PREFIX)]

    geometry = present(
        [
            "geom_x",
            "geom_y",
            "density_score",
            "mean_knn_observable_distance",
        ]
    )

    phase_dynamic = present(
        [
            "dyn__signed_phase",
            "dyn__distance_to_seam",
            "dyn__grad_signed_phase_norm",
        ]
    )

    phase_mixed = present(
        [
            "dyn__signed_phase",
            "dyn__distance_to_seam",
            "dyn__grad_signed_phase_norm",
            "signed_phase",
            "distance_to_seam",
            "phase_contrast",
            "seam_proxy_score",
            "is_seam_proxy",
        ]
    )

    response_lazarus_dynamic = present(
        [
            "dyn__response_strength",
            "dyn__frobenius_T",
            "dyn__trace_T",
            "dyn__lazarus_score",
            "dyn__grad_lazarus_score_norm",
        ]
    )

    response_lazarus_mixed = present(
        [
            "dyn__response_strength",
            "dyn__frobenius_T",
            "dyn__trace_T",
            "dyn__lazarus_score",
            "dyn__grad_lazarus_score_norm",
            "response_strength",
            "frobenius_T",
            "trace_T",
            "lazarus_score",
        ]
    )

    coupling_dynamic = present(
        [
            "dyn__signed_coupling",
            "dyn__cosine_alignment",
        ]
    )

    coupling_mixed = present(
        [
            "dyn__signed_coupling",
            "dyn__cosine_alignment",
            "signed_coupling",
            "cosine_alignment",
        ]
    )

    tensor_dynamic = present(
        [
            "dyn__T_xx",
            "dyn__T_xy",
            "dyn__T_yx",
            "dyn__T_yy",
            "dyn__trace_T",
            "dyn__frobenius_T",
            "dyn__response_strength",
        ]
    )

    seam_density_energy = present(
        [
            "energy",
            "density_score",
            "seam_proxy_score",
            "is_seam_proxy",
            "phase_contrast",
            "mean_knn_observable_distance",
        ]
    )

    semantic_phase = phase_dynamic if dynamic_only_families else phase_mixed
    semantic_response = response_lazarus_dynamic if dynamic_only_families else response_lazarus_mixed
    semantic_coupling = coupling_dynamic if dynamic_only_families else coupling_mixed

    families = {
        "diffused_observables": dyn_cols,
        "geometry": geometry,
        "phase": semantic_phase,
        "phase_dynamic": phase_dynamic,
        "response_lazarus": semantic_response,
        "response_lazarus_dynamic": response_lazarus_dynamic,
        "coupling": semantic_coupling,
        "coupling_dynamic": coupling_dynamic,
        "tensor_dynamic": tensor_dynamic,
        "seam_density_energy": seam_density_energy,
        "object_memberships": obj_cols,
        "all": cols,
    }

    if dynamic_only_families:
        dynamic_allowed = set(dyn_cols) | set(obj_cols) | set(geometry) | set(seam_density_energy)
        families["all_dynamic_plus_proxies"] = [c for c in cols if c in dynamic_allowed]
        families["all_dynamic_only"] = dyn_cols

    return families


def build_model(model_family: str, random_state: int):
    if model_family == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_family == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        )

    raise ValueError(f"Unknown model_family: {model_family}")


def cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    model_family: str,
    cv_splits: int,
    random_state: int,
) -> dict:
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_splits = min(cv_splits, int(counts.min()))

    if len(classes) < 2 or n_splits < 2:
        return {
            "status": "insufficient_classes_or_splits",
            "balanced_accuracy_mean": np.nan,
            "balanced_accuracy_std": np.nan,
            "roc_auc_mean": np.nan,
            "roc_auc_std": np.nan,
            "n_splits": n_splits,
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_model = build_model(model_family, random_state)

    ba_scores = []
    auc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        model = clone(base_model)
        model.fit(X[train_idx], y[train_idx])

        pred = model.predict(X[test_idx])
        ba_scores.append(balanced_accuracy_score(y[test_idx], pred))

        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X[test_idx])
                if len(classes) == 2:
                    auc = roc_auc_score(y[test_idx], proba[:, 1])
                else:
                    auc = roc_auc_score(y[test_idx], proba, multi_class="ovr")
                auc_scores.append(auc)
        except Exception:
            pass

    return {
        "status": "ok",
        "balanced_accuracy_mean": float(np.mean(ba_scores)),
        "balanced_accuracy_std": float(np.std(ba_scores)),
        "roc_auc_mean": float(np.mean(auc_scores)) if auc_scores else np.nan,
        "roc_auc_std": float(np.std(auc_scores)) if auc_scores else np.nan,
        "n_splits": int(n_splits),
    }


def impute_matrix(x_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    # pandas may hand NumPy a read-only view; force a writable copy.
    x = np.asarray(
        x_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        dtype=float,
    ).copy()

    imputed_counts = {}

    for j, col in enumerate(x_df.columns):
        finite = np.isfinite(x[:, j])
        n_missing = int((~finite).sum())
        imputed_counts[col] = n_missing

        if finite.any():
            med = float(np.nanmedian(x[finite, j]))
        else:
            med = 0.0

        x[~finite, j] = med

    return x, imputed_counts


def extract_coefficients(
    df_scale: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    model_family: str,
    random_state: int,
    top_n: int,
    family: str,
    scale_index: int,
    t: float,
) -> list[dict]:
    if model_family != "logreg":
        return []

    x_df = df_scale[feature_cols]
    x, _ = impute_matrix(x_df)

    model = build_model(model_family, random_state)
    model.fit(x, y)

    try:
        clf = model.named_steps["clf"]
        coefs = clf.coef_
    except Exception:
        return []

    if coefs.ndim != 2:
        return []

    rows = []

    if coefs.shape[0] == 1:
        coef = coefs[0]
        order = np.argsort(np.abs(coef))[::-1][:top_n]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "scale_index": scale_index,
                    "t": t,
                    "feature_family": family,
                    "rank": rank,
                    "class_index": 1,
                    "feature": feature_cols[idx],
                    "feature_source_kind": feature_source_kind(feature_cols[idx]),
                    "coefficient": float(coef[idx]),
                    "abs_coefficient": float(abs(coef[idx])),
                }
            )
        return rows

    for class_index in range(coefs.shape[0]):
        coef = coefs[class_index]
        order = np.argsort(np.abs(coef))[::-1][:top_n]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "scale_index": scale_index,
                    "t": t,
                    "feature_family": family,
                    "rank": rank,
                    "class_index": class_index,
                    "feature": feature_cols[idx],
                    "feature_source_kind": feature_source_kind(feature_cols[idx]),
                    "coefficient": float(coef[idx]),
                    "abs_coefficient": float(abs(coef[idx])),
                }
            )

    return rows


def run_separability(
    cfg: Config,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_names = sorted(df["case"].unique())
    case_to_code = {name: i for i, name in enumerate(case_names)}

    df = df.copy()
    df["case_code"] = df["case"].map(case_to_code).astype(int)

    families = family_columns(df, dynamic_only_families=cfg.dynamic_only_families)

    feature_manifest_rows = []
    for fam, cols in families.items():
        source_counts = {}
        for c in cols:
            k = feature_source_kind(c)
            source_counts[k] = source_counts.get(k, 0) + 1

        feature_manifest_rows.append(
            {
                "feature_family": fam,
                "n_features": len(cols),
                "source_counts": json.dumps(source_counts, sort_keys=True),
                "features": ",".join(cols),
                "status": "ok" if cols else "no_features",
            }
        )

    scores = []
    coef_rows = []

    for scale_index, df_scale in df.groupby("scale_index", sort=True):
        t = float(df_scale["t"].iloc[0])
        y = df_scale["case_code"].to_numpy(dtype=int)
        labels, counts = np.unique(y, return_counts=True)

        for fam, cols in families.items():
            source_counts = {}
            for c in cols:
                k = feature_source_kind(c)
                source_counts[k] = source_counts.get(k, 0) + 1

            row_base = {
                "scale_index": int(scale_index),
                "t": t,
                "feature_family": fam,
                "model_family": cfg.model_family,
                "dynamic_only_families": int(cfg.dynamic_only_families),
                "n_samples": int(len(df_scale)),
                "n_cases": int(len(labels)),
                "case_names": ",".join(case_names),
                "class_counts": json.dumps(
                    {case_names[int(k)]: int(v) for k, v in zip(labels, counts)},
                    sort_keys=True,
                ),
                "n_features": int(len(cols)),
                "source_counts": json.dumps(source_counts, sort_keys=True),
            }

            if not cols:
                scores.append(
                    {
                        **row_base,
                        "status": "no_features",
                        "balanced_accuracy_mean": np.nan,
                        "balanced_accuracy_std": np.nan,
                        "roc_auc_mean": np.nan,
                        "roc_auc_std": np.nan,
                        "n_splits": 0,
                    }
                )
                continue

            if counts.min() < cfg.min_class_count:
                scores.append(
                    {
                        **row_base,
                        "status": "below_min_class_count",
                        "balanced_accuracy_mean": np.nan,
                        "balanced_accuracy_std": np.nan,
                        "roc_auc_mean": np.nan,
                        "roc_auc_std": np.nan,
                        "n_splits": 0,
                    }
                )
                continue

            x_df = df_scale[cols]
            x, _ = impute_matrix(x_df)

            try:
                res = cv_scores(
                    X=x,
                    y=y,
                    model_family=cfg.model_family,
                    cv_splits=cfg.cv_splits,
                    random_state=cfg.random_state,
                )
                scores.append({**row_base, **res})

                if res["status"] == "ok":
                    coef_rows.extend(
                        extract_coefficients(
                            df_scale=df_scale,
                            feature_cols=cols,
                            y=y,
                            model_family=cfg.model_family,
                            random_state=cfg.random_state,
                            top_n=cfg.top_coefficients,
                            family=fam,
                            scale_index=int(scale_index),
                            t=t,
                        )
                    )
            except Exception as exc:
                scores.append(
                    {
                        **row_base,
                        "status": f"fit_or_eval_error:{type(exc).__name__}:{exc}",
                        "balanced_accuracy_mean": np.nan,
                        "balanced_accuracy_std": np.nan,
                        "roc_auc_mean": np.nan,
                        "roc_auc_std": np.nan,
                        "n_splits": 0,
                    }
                )

    return pd.DataFrame(scores), pd.DataFrame(coef_rows), pd.DataFrame(feature_manifest_rows)


def write_report(
    cfg: Config,
    input_manifest: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    scores: pd.DataFrame,
    coefs: pd.DataFrame,
) -> None:
    lines = [
        "# OBS-076d — Scale-conditioned factor separability",
        "",
        "## Scope",
        "",
        "OBS-076d tests case separability across diffusion scale using structural-factor feature families.",
        "",
        "This is node-level corpus/case separability, not path-label transfer asymmetry.",
        "",
        "## v2 patch",
        "",
        "v2 adds dynamic-only family support and explicit dynamic semantic families.",
        "",
        f"- dynamic_only_families: `{cfg.dynamic_only_families}`",
        "",
        "## Configuration",
        "",
        f"- model_family: `{cfg.model_family}`",
        f"- cv_splits: `{cfg.cv_splits}`",
        f"- random_state: `{cfg.random_state}`",
        f"- min_class_count: `{cfg.min_class_count}`",
        f"- top_coefficients: `{cfg.top_coefficients}`",
        "",
        "## Inputs",
        "",
        "| case | artifact | status | details | path |",
        "| --- | --- | --- | --- | --- |",
    ]

    for row in input_manifest.itertuples(index=False):
        lines.append(
            f"| {row.case} | {row.artifact} | {row.status} | {row.details} | `{row.path}` |"
        )

    lines.extend(
        [
            "",
            "## Feature families",
            "",
            "| feature_family | n_features | source_counts | status |",
            "| --- | ---: | --- | --- |",
        ]
    )

    for row in feature_manifest.sort_values("feature_family").itertuples(index=False):
        lines.append(
            f"| {row.feature_family} | {int(row.n_features)} | `{row.source_counts}` | {row.status} |"
        )

    lines.extend(
        [
            "",
            "## Separability scores",
            "",
            "| scale_index | t | feature_family | n_features | source_counts | status | balanced_accuracy | roc_auc |",
            "| ---: | ---: | --- | ---: | --- | --- | ---: | ---: |",
        ]
    )

    display = scores.sort_values(["scale_index", "feature_family"])
    for row in display.itertuples(index=False):
        ba = "" if pd.isna(row.balanced_accuracy_mean) else f"{float(row.balanced_accuracy_mean):.6g}"
        auc = "" if pd.isna(row.roc_auc_mean) else f"{float(row.roc_auc_mean):.6g}"
        lines.append(
            "| "
            f"{int(row.scale_index)} | "
            f"{float(row.t):.6g} | "
            f"{row.feature_family} | "
            f"{int(row.n_features)} | "
            f"`{row.source_counts}` | "
            f"{row.status} | "
            f"{ba} | "
            f"{auc} |"
        )

    lines.extend(["", "## Best family by scale", ""])

    ok = scores[scores["status"] == "ok"].copy()
    if ok.empty:
        lines.append("No successful separability scores.")
    else:
        best = (
            ok.sort_values(["scale_index", "balanced_accuracy_mean"], ascending=[True, False])
            .groupby("scale_index", as_index=False)
            .head(1)
        )
        lines.extend(
            [
                "| scale_index | t | feature_family | balanced_accuracy | roc_auc |",
                "| ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in best.itertuples(index=False):
            auc = "" if pd.isna(row.roc_auc_mean) else f"{float(row.roc_auc_mean):.6g}"
            lines.append(
                "| "
                f"{int(row.scale_index)} | "
                f"{float(row.t):.6g} | "
                f"{row.feature_family} | "
                f"{float(row.balanced_accuracy_mean):.6g} | "
                f"{auc} |"
            )

    if not coefs.empty:
        lines.extend(["", "## Top coefficients at final scale", ""])
        final_scale = int(coefs["scale_index"].max())
        final = coefs[coefs["scale_index"] == final_scale].copy()
        final = final.sort_values(["feature_family", "rank"]).groupby("feature_family").head(8)

        lines.extend(
            [
                "| feature_family | rank | feature | source_kind | coefficient | abs_coefficient |",
                "| --- | ---: | --- | --- | ---: | ---: |",
            ]
        )
        for row in final.itertuples(index=False):
            lines.append(
                "| "
                f"{row.feature_family} | "
                f"{int(row.rank)} | "
                f"{row.feature} | "
                f"{row.feature_source_kind} | "
                f"{float(row.coefficient):.6g} | "
                f"{float(row.abs_coefficient):.6g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This is scale-conditioned case separability, not Cp3→Cp2 transfer asymmetry.",
            "- Features are node-level observable-space and structural-proxy features.",
            "- Dynamic families use injected OBS-076a `dyn__` columns.",
            "- Dynamic-proxy families use OBS-076b scale-dependent geometry/proxy fields.",
            "- High separability means cases occupy different scale-conditioned supports.",
            "- Coefficients are diagnostic, not causal.",
            "- Later OBS studies should connect this to path-level labels and transfer asymmetry.",
            "",
            "## Output artifacts",
            "",
            "- `obs076d_input_manifest.csv`",
            "- `obs076d_feature_manifest.csv`",
            "- `obs076d_scale_feature_table.csv`",
            "- `obs076d_separability_scores.csv`",
            "- `obs076d_feature_coefficients.csv`",
            "- `obs076d_report.md`",
            "",
        ]
    )

    (cfg.outdir / "obs076d_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    df, input_manifest = load_all_cases(cfg)
    scores, coefs, feature_manifest = run_separability(cfg, df)

    input_manifest.to_csv(cfg.outdir / "obs076d_input_manifest.csv", index=False)
    feature_manifest.to_csv(cfg.outdir / "obs076d_feature_manifest.csv", index=False)
    df.to_csv(cfg.outdir / "obs076d_scale_feature_table.csv", index=False)
    scores.to_csv(cfg.outdir / "obs076d_separability_scores.csv", index=False)
    coefs.to_csv(cfg.outdir / "obs076d_feature_coefficients.csv", index=False)

    write_report(
        cfg=cfg,
        input_manifest=input_manifest,
        feature_manifest=feature_manifest,
        scores=scores,
        coefs=coefs,
    )

    print("OBS-076d complete")
    print("wrote:", cfg.outdir / "obs076d_input_manifest.csv")
    print("wrote:", cfg.outdir / "obs076d_feature_manifest.csv")
    print("wrote:", cfg.outdir / "obs076d_scale_feature_table.csv")
    print("wrote:", cfg.outdir / "obs076d_separability_scores.csv")
    print("wrote:", cfg.outdir / "obs076d_feature_coefficients.csv")
    print("wrote:", cfg.outdir / "obs076d_report.md")


if __name__ == "__main__":
    main()
