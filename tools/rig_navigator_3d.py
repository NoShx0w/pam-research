#!/usr/bin/env python3
"""
rig_navigator_3d.py (v2)

Local-first RIG Navigator prototype for the PAM Observatory.

v2 purpose
----------
RIG Navigator v1 was an artifact viewer.
RIG Navigator v2 adds a derived geometry layer over the OBS-078a stability-core
feature table and OBS-081 relation × carrier registry.

Generated outputs
-----------------
Always:
  outputs/rig_navigator/views/rig_navigator_index.html
  outputs/rig_navigator/views/rig_navigator_manifest.csv
  outputs/rig_navigator/views/rig_stability_core_geometry.csv
  outputs/rig_navigator/views/rig_registry_plot_points.csv

When Plotly is available and data exists:
  outputs/rig_navigator/views/stability_core_3d.html
  outputs/rig_navigator/views/rig_registry_3d.html

Demo mode
---------
--demo creates clearly labeled mock data:
DEMO / MOCK DATA — NOT OBSERVATORY EVIDENCE

No requirements.txt changes are needed. Plotly is imported lazily.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError as exc:
    pd = None  # type: ignore[assignment]
    PANDAS_IMPORT_ERROR = exc
else:
    PANDAS_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:
    np = None  # type: ignore[assignment]
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

DEMO_BANNER = "DEMO / MOCK DATA — NOT OBSERVATORY EVIDENCE"

DEFAULT_REGISTRY = Path("outputs/rig_registry/rig_relation_registry.csv")
DEFAULT_SURVIVAL = Path("outputs/rig_registry/rig_survival_matrix.csv")
DEFAULT_FEATURE_TABLE = Path(
    "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv"
)
DEFAULT_OUTDIR = Path("outputs/rig_navigator/views")

STABILITY_CORE_COLUMNS = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

GEOMETRY_COLUMNS = [
    "row_id",
    "case",
    "object",
    "cohort",
    "scale_index_from",
    "scale_index_to",
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
    "divergence_mean",
    "unboundedness",
    "instability_signature",
    "bounded_stability_signature",
    "local_density",
    "local_case_purity",
    "local_boundary_score",
    "distance_to_case_centroid",
    "distance_to_global_centroid",
    "local_instability_mean",
    "local_bounded_stability_mean",
    "local_repair_pressure",
    "nearest_case_margin",
]

REGISTRY_PLOT_COLUMNS = [
    "relation_id",
    "task",
    "carrier",
    "carrier_role",
    "rig_status",
    "x_task_index",
    "y_carrier_index",
    "z_survival",
    "obs080c_carrier_ba",
    "obs080d_carrier_mean_ba",
    "obs080d_carrier_min_ci95_low",
    "task_geometry_needed_level",
    "repair_recommendation",
]

STATUS_COLORS = {
    "stable_reusable_invariant": "#65ff7a",
    "context_sensitive_reusable_invariant": "#39e7ff",
    "redundant_reusable_invariant": "#b16cff",
    "weak_redundant_carrier": "#ffad33",
    "fragile_candidate": "#ff5c7a",
    "accidental_relation": "#777777",
    "insufficient_evidence": "#aaaaaa",
}

CASE_COLORS = {
    "C": "#65ff7a",
    "Cp2": "#ffad33",
    "Cp3": "#b16cff",
}


@dataclass
class ArtifactStatus:
    label: str
    path: Path
    required_for: str
    exists: bool = False
    rows: int | None = None
    columns: int | None = None
    status: str = "missing"
    message: str = ""


@dataclass
class Metrics:
    feature_table_rows: int = 0
    valid_core_rows: int = 0
    registry_rows: int = 0
    knn_k: int = 12
    knn_enabled: bool = False
    transition_lines_enabled: bool = False
    transition_line_count: int = 0
    registry_overlay_enabled: bool = False
    plotly_available: bool = False
    demo_mode: bool = False


@dataclass
class RunState:
    generated: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    plotly_hint: str = "Install Plotly to render interactive 3D views: pip install plotly"
    demo: bool = False
    metrics: Metrics = field(default_factory=Metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local RIG Navigator v2 geometry/HTML views from OBS-081 / OBS-078a CSV artifacts."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Path to rig_relation_registry.csv")
    parser.add_argument("--survival", type=Path, default=DEFAULT_SURVIVAL, help="Path to rig_survival_matrix.csv")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE, help="Path to obs078a_feature_table.csv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for generated HTML/CSV views")
    parser.add_argument("--demo", action="store_true", help="Use clearly marked mock data when artifacts are missing")
    parser.add_argument("--open", action="store_true", dest="open_index", help="Open the generated index in the default browser")
    parser.add_argument("--knn-k", type=int, default=12, help="k for local nearest-neighbor stability geometry")
    parser.add_argument("--no-transition-lines", action="store_true", help="Disable transition line traces in stability_core_3d.html")
    parser.add_argument("--no-registry-overlay", action="store_true", help="Disable secondary registry annotations in stability_core_3d.html")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def import_plotly() -> tuple[Any | None, Any | None, str | None]:
    try:
        import plotly.express as px  # type: ignore
        import plotly.graph_objects as go  # type: ignore
    except ImportError as exc:
        return None, None, str(exc)
    return px, go, None


def load_csv(path: Path, label: str, required_for: str, state: RunState) -> tuple[Any | None, ArtifactStatus]:
    status = ArtifactStatus(label=label, path=path, required_for=required_for, exists=path.exists())
    if pd is None:
        status.status = "error"
        status.message = f"pandas is unavailable: {PANDAS_IMPORT_ERROR}"
        state.errors.append(status.message)
        return None, status
    if not path.exists():
        status.status = "missing"
        status.message = "expected but absent from repository checkout or local outputs"
        return None, status
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        status.status = "error"
        status.message = f"failed to read CSV: {exc}"
        state.errors.append(f"{label}: {status.message}")
        return None, status
    status.rows = int(len(df))
    status.columns = int(len(df.columns))
    status.status = "loaded"
    status.message = "loaded"
    return df, status


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    available = set(cols)
    for name in candidates:
        if name in available:
            return name
    lower_map = {c.lower(): c for c in cols}
    for name in candidates:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def numeric_series(df: Any, column: str) -> Any:
    return pd.to_numeric(df[column], errors="coerce")


def require_columns(df: Any, required: list[str]) -> list[str]:
    return [col for col in required if col not in list(df.columns)]


def make_demo_feature_table() -> Any:
    rows = []
    cases = ["C", "Cp2", "Cp3"]
    objects = ["density_core", "seam_corridor", "transition_bridge", "settlement_band"]
    for i in range(72):
        case = cases[i % len(cases)]
        phase = i / 8.0
        rows.append(
            {
                "case": case,
                "candidate_rank": (i % 6) + 1,
                "object": objects[i % len(objects)],
                "scale_index_from": 4 + (i % 9),
                "scale_index_to": 5 + (i % 9),
                "cohort": "after" if i % 2 else "before",
                "dominant_family": ["intrinsic_dimension", "boundedness", "transition_geometry"][i % 3],
                "dominant_reason": ["local_mle_delta", "bounded_share", "delta_drift"][i % 3],
                "mean_lambda_local_mean": round(math.sin(phase) + {"C": -0.55, "Cp2": 0.15, "Cp3": 0.65}[case], 4),
                "mean_delta_d_mean": round(math.cos(phase * 0.9) + {"C": -0.15, "Cp2": 0.75, "Cp3": 0.35}[case], 4),
                "bounded_share_mean": round(max(0.02, min(0.98, {"C": 0.75, "Cp2": 0.28, "Cp3": 0.18}[case] + 0.12 * math.sin(phase * 1.7))), 4),
                "n_paths": 1000 + i * 37,
                "demo_notice": DEMO_BANNER,
            }
        )
    return pd.DataFrame(rows)


def make_demo_registry() -> Any:
    records = [
        ("C_vs_Cp2", "stability_core_3", "compact_core_carrier", "stable_reusable_invariant", 0.984, 0.958, 0.983, 0.960, 23, "Level 1", "compact core sufficient", "preserve compact core; no repair needed"),
        ("C_vs_Cp3", "stability_core_3", "compact_core_carrier", "stable_reusable_invariant", 0.989, 0.970, 0.986, 0.972, 23, "Level 1", "compact core sufficient", "preserve compact core; no repair needed"),
        ("Cp2_vs_Cp3", "stability_core_3", "compact_core_carrier", "context_sensitive_reusable_invariant", 0.812, 0.744, 0.801, 0.720, 23, "Level 3", "geometry sharpening useful", "prefer geometry sharpening carrier"),
        ("three_way", "stability_core_3", "compact_core_carrier", "context_sensitive_reusable_invariant", 0.894, 0.838, 0.872, 0.812, 23, "Level 3", "enriched geometry improves precision", "annotate structural sensitivity"),
        ("Cp2_vs_Cp3", "geometry_scores_only", "geometry_sharpening_carrier", "weak_redundant_carrier", 0.694, 0.612, 0.676, 0.590, 14, "Level 4", "path support needed", "repair with stability-plus-geometry"),
        ("three_way", "path_shares_only", "path_support_carrier", "fragile_candidate", 0.621, 0.553, 0.604, 0.520, 14, "Level 5", "strict context needed", "do not promote without additional evidence"),
    ]
    cols = [
        "task", "carrier", "carrier_role", "rig_status", "mean_survival_score", "min_survival_score",
        "obs080d_carrier_mean_ba", "obs080d_carrier_min_ci95_low", "n_survival_rows",
        "task_geometry_needed_level", "task_geometry_needed_label", "repair_recommendation",
    ]
    df = pd.DataFrame(records, columns=cols)
    df["relation_id"] = df["task"] + "__" + df["carrier"]
    df["obs080c_carrier_ba"] = df["mean_survival_score"]
    df["demo_notice"] = DEMO_BANNER
    return df


def empty_geometry_frame() -> Any:
    return pd.DataFrame(columns=GEOMETRY_COLUMNS)


def empty_registry_plot_frame() -> Any:
    return pd.DataFrame(columns=REGISTRY_PLOT_COLUMNS)


def add_nan_columns(df: Any, columns: list[str]) -> Any:
    for col in columns:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def compute_stability_geometry(feature_df: Any | None, outdir: Path, state: RunState, k: int) -> Any:
    path = outdir / "rig_stability_core_geometry.csv"
    if feature_df is None or pd is None:
        geom = empty_geometry_frame()
        geom.to_csv(path, index=False)
        state.generated.append(path)
        state.warnings.append("rig_stability_core_geometry.csv written empty: feature table not available")
        return geom

    state.metrics.feature_table_rows = int(len(feature_df))
    missing = require_columns(feature_df, STABILITY_CORE_COLUMNS)
    if missing:
        geom = empty_geometry_frame()
        geom.to_csv(path, index=False)
        state.generated.append(path)
        state.warnings.append("rig_stability_core_geometry.csv written empty: missing required core columns " + ", ".join(missing))
        return geom

    df = feature_df.copy()
    df.insert(0, "row_id", range(len(df))) if "row_id" not in df.columns else None
    for col in STABILITY_CORE_COLUMNS:
        df[col] = numeric_series(df, col)

    valid_mask = df[STABILITY_CORE_COLUMNS].notna().all(axis=1)
    state.metrics.valid_core_rows = int(valid_mask.sum())

    df["divergence_mean"] = df[["mean_lambda_local_mean", "mean_delta_d_mean"]].mean(axis=1)
    df["unboundedness"] = 1.0 - df["bounded_share_mean"]
    df["instability_signature"] = df[["mean_lambda_local_mean", "mean_delta_d_mean", "unboundedness"]].mean(axis=1)
    df["bounded_stability_signature"] = df["bounded_share_mean"] - df["divergence_mean"]

    knn_cols = [
        "local_density", "local_case_purity", "local_boundary_score", "distance_to_case_centroid",
        "distance_to_global_centroid", "local_instability_mean", "local_bounded_stability_mean",
        "local_repair_pressure", "nearest_case_margin",
    ]
    add_nan_columns(df, knn_cols)

    if np is None:
        state.warnings.append(f"kNN geometry skipped: numpy unavailable ({NUMPY_IMPORT_ERROR}).")
    elif k < 1:
        state.warnings.append("kNN geometry skipped: --knn-k must be at least 1.")
    elif state.metrics.valid_core_rows < k + 1:
        state.warnings.append("kNN geometry skipped: fewer than k + 1 valid core rows.")
    else:
        valid_idx = df.index[valid_mask].to_list()
        coords = df.loc[valid_idx, STABILITY_CORE_COLUMNS].to_numpy(dtype=float)
        cases = df.loc[valid_idx, "case"].astype(str).to_numpy() if "case" in df.columns else None
        inst = df.loc[valid_idx, "instability_signature"].to_numpy(dtype=float)
        bst = df.loc[valid_idx, "bounded_stability_signature"].to_numpy(dtype=float)
        global_centroid = coords.mean(axis=0)

        if cases is not None:
            case_centroids = {}
            for case in sorted(set(cases)):
                case_centroids[case] = coords[cases == case].mean(axis=0)
        else:
            case_centroids = {}

        n = len(coords)
        for local_i, idx in enumerate(valid_idx):
            delta = coords - coords[local_i]
            distances = np.linalg.norm(delta, axis=1)
            order = np.argsort(distances)
            neighbor_order = [j for j in order if j != local_i][:k]
            neighbor_distances = distances[neighbor_order]
            mean_distance = float(np.mean(neighbor_distances)) if len(neighbor_distances) else float("nan")
            df.at[idx, "local_density"] = 1.0 / (mean_distance + 1e-12) if math.isfinite(mean_distance) else float("nan")
            df.at[idx, "distance_to_global_centroid"] = float(np.linalg.norm(coords[local_i] - global_centroid))
            df.at[idx, "local_instability_mean"] = float(np.nanmean(inst[neighbor_order]))
            df.at[idx, "local_bounded_stability_mean"] = float(np.nanmean(bst[neighbor_order]))

            if cases is not None:
                same = cases[neighbor_order] == cases[local_i]
                purity = float(np.mean(same)) if len(same) else float("nan")
                df.at[idx, "local_case_purity"] = purity
                df.at[idx, "local_boundary_score"] = 1.0 - purity if math.isfinite(purity) else float("nan")
                df.at[idx, "distance_to_case_centroid"] = float(np.linalg.norm(coords[local_i] - case_centroids[cases[local_i]]))
                other_dist = distances[(cases != cases[local_i])]
                same_dist = distances[(cases == cases[local_i]) & (distances > 0)]
                nearest_other = float(np.min(other_dist)) if len(other_dist) else float("nan")
                nearest_same = float(np.min(same_dist)) if len(same_dist) else float("nan")
                if math.isfinite(nearest_other) and math.isfinite(nearest_same):
                    df.at[idx, "nearest_case_margin"] = nearest_other - nearest_same
                elif math.isfinite(nearest_other):
                    df.at[idx, "nearest_case_margin"] = nearest_other
            else:
                df.at[idx, "local_case_purity"] = float("nan")
                df.at[idx, "local_boundary_score"] = float("nan")
                df.at[idx, "distance_to_case_centroid"] = float("nan")

            boundary = df.at[idx, "local_boundary_score"]
            bounded_stability = df.at[idx, "bounded_stability_signature"]
            if pd.notna(boundary) and pd.notna(bounded_stability):
                df.at[idx, "local_repair_pressure"] = float(boundary) * max(0.0, -float(bounded_stability))
        state.metrics.knn_enabled = True

    preferred_cols = [c for c in GEOMETRY_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in preferred_cols]
    geom = df[preferred_cols + extra_cols]
    geom.to_csv(path, index=False)
    state.generated.append(path)
    return geom


def category_codes(series: Any) -> tuple[Any, dict[str, int]]:
    labels = [str(v) for v in series.fillna("unknown").tolist()]
    ordered = list(dict.fromkeys(labels))
    mapping = {label: i for i, label in enumerate(ordered)}
    return pd.Series(labels, index=series.index).map(mapping), mapping


def compute_registry_plot_points(registry_df: Any | None, outdir: Path, state: RunState) -> Any:
    path = outdir / "rig_registry_plot_points.csv"
    if registry_df is None or pd is None:
        points = empty_registry_plot_frame()
        points.to_csv(path, index=False)
        state.generated.append(path)
        state.warnings.append("rig_registry_plot_points.csv written empty: registry artifact absent.")
        return points

    df = registry_df.copy()
    state.metrics.registry_rows = int(len(df))
    task_col = pick_column(df.columns, ["task", "relation", "relation_task"])
    carrier_col = pick_column(df.columns, ["carrier", "feature_contract", "contract", "contract_name"])
    carrier_role_col = pick_column(df.columns, ["carrier_role", "role"])
    status_col = pick_column(df.columns, ["rig_status", "status"])

    if task_col is None:
        df["task"] = "unknown_task"
        task_col = "task"
        state.warnings.append("registry plot points: missing task column; using unknown_task.")
    if carrier_col is None:
        df["carrier"] = df[carrier_role_col] if carrier_role_col else "unknown_carrier"
        carrier_col = "carrier"
        state.warnings.append("registry plot points: missing carrier/feature_contract column; using fallback carrier.")
    if carrier_role_col is None:
        df["carrier_role"] = df[carrier_col]
        carrier_role_col = "carrier_role"
    if status_col is None:
        df["rig_status"] = "unknown_status"
        status_col = "rig_status"
        state.warnings.append("registry plot points: missing rig_status/status column; using unknown_status.")

    if "relation_id" not in df.columns:
        df["relation_id"] = df[task_col].astype(str) + "__" + df[carrier_col].astype(str)

    for col in ["obs080c_carrier_ba", "obs080d_carrier_mean_ba", "obs080d_carrier_min_ci95_low", "mean_survival_score"]:
        if col in df.columns:
            df[col] = numeric_series(df, col)

    z_source = pick_column(df.columns, ["obs080d_carrier_mean_ba", "mean_survival_score", "obs080c_carrier_ba"])
    if z_source is None:
        df["z_survival"] = float("nan")
        state.warnings.append("registry plot points: z_survival unavailable; no obs080d_carrier_mean_ba, mean_survival_score, or obs080c_carrier_ba column.")
    else:
        df["z_survival"] = numeric_series(df, z_source)

    df["x_task_index"], task_map = category_codes(df[task_col])
    df["y_carrier_index"], carrier_map = category_codes(df[carrier_role_col])
    df["task"] = df[task_col].astype(str)
    df["carrier"] = df[carrier_col].astype(str)
    df["carrier_role"] = df[carrier_role_col].astype(str)
    df["rig_status"] = df[status_col].astype(str)

    for col in REGISTRY_PLOT_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan") if col.startswith("obs") or col in {"z_survival"} else ""
    points = df[REGISTRY_PLOT_COLUMNS + [c for c in df.columns if c not in REGISTRY_PLOT_COLUMNS]]
    points.attrs["task_map"] = task_map
    points.attrs["carrier_map"] = carrier_map
    points.to_csv(path, index=False)
    state.generated.append(path)
    return points


def html_shell(title: str, body: str, demo: bool = False) -> str:
    banner = f"<div class='demo'>{html.escape(DEMO_BANNER)}</div>" if demo else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #05080b; --panel: #0b1117; --panel2: #101820; --line: #23303a;
  --text: #d8e3e7; --muted: #85939c; --green: #65ff7a; --cyan: #39e7ff;
  --purple: #b16cff; --orange: #ffad33; --red: #ff5c7a;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: radial-gradient(circle at 50% 20%, #101827 0, #05080b 45%, #020304 100%); color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; }}
a {{ color: var(--cyan); text-decoration: none; }} a:hover {{ text-decoration: underline; }}
.header {{ display:flex; align-items:center; justify-content:space-between; padding:14px 18px; border-bottom:1px solid var(--line); background:rgba(5,8,11,.92); position:sticky; top:0; z-index:10; }}
.title {{ font-size:18px; font-weight:700; letter-spacing:.02em; }}
.live {{ color:var(--green); border:1px solid #2a7c39; padding:3px 7px; border-radius:3px; margin-left:10px; font-size:12px; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px; }}
.panel {{ background:rgba(11,17,23,.92); border:1px solid var(--line); border-radius:7px; overflow:hidden; box-shadow:0 0 30px rgba(0,0,0,.35); }}
.panel h2 {{ margin:0; padding:10px 12px; font-size:13px; border-bottom:1px solid var(--line); color:#f2fbff; text-transform:uppercase; letter-spacing:.06em; }}
.panel h3 {{ font-size:12px; color: var(--cyan); margin: 12px 0 6px; }}
.panel .inner {{ padding:12px; }}
.console {{ margin:0 12px 12px; background:#06090d; border:1px solid var(--line); border-radius:7px; padding:12px; color:var(--muted); }}
.badge {{ display:inline-block; padding:3px 7px; border-radius:4px; border:1px solid var(--line); margin:2px 4px 2px 0; }}
.good {{ color:var(--green); border-color:#2a7c39; }} .warn {{ color:var(--orange); border-color:#8a641f; }} .bad {{ color:var(--red); border-color:#813040; }}
.muted {{ color:var(--muted); }} .cyan {{ color:var(--cyan); }} .green {{ color:var(--green); }} .orange {{ color:var(--orange); }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }} th, td {{ padding:7px 8px; border-bottom:1px solid #17232c; text-align:left; vertical-align:top; }} th {{ color:#aebbc2; background:#0c131a; }}
pre {{ white-space:pre-wrap; background:#05080b; border:1px solid var(--line); padding:10px; border-radius:6px; color:#bfccd2; }}
.demo {{ margin:12px; border:1px solid var(--orange); color:var(--orange); background:rgba(255,173,51,.08); padding:10px 12px; border-radius:6px; font-weight:700; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="header"><div class="title">PAM Observatory — RIG Navigator v2 <span class="live">LOCAL</span></div><div class="muted">generated {html.escape(utc_now())}</div></div>
{banner}
{body}
</body>
</html>"""


def dataframe_preview(df: Any, max_rows: int = 8) -> str:
    if df is None or len(df) == 0:
        return "<p class='muted'>No rows available.</p>"
    cols = list(df.columns)[:8]
    rows = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead><tbody>"]
    for _, rec in df[cols].head(max_rows).iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(rec.get(c, '')))}</td>" for c in cols) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def status_table(statuses: list[ArtifactStatus]) -> str:
    rows = ["<table><thead><tr><th>artifact</th><th>path</th><th>required for</th><th>status</th><th>rows</th><th>message</th></tr></thead><tbody>"]
    for s in statuses:
        cls = "good" if s.status in {"loaded", "demo"} else ("bad" if s.status == "error" else "warn")
        rows.append(
            "<tr>"
            f"<td>{html.escape(s.label)}</td>"
            f"<td><code>{html.escape(repo_rel(s.path))}</code></td>"
            f"<td>{html.escape(s.required_for)}</td>"
            f"<td><span class='badge {cls}'>{html.escape(s.status)}</span></td>"
            f"<td>{'' if s.rows is None else s.rows}</td>"
            f"<td>{html.escape(s.message)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def write_manifest(outdir: Path, statuses: list[ArtifactStatus], state: RunState) -> Path:
    manifest = outdir / "rig_navigator_manifest.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "generated_at_utc", "record_type", "artifact", "path", "required_for", "exists", "status", "rows", "columns", "message",
        "feature_table_rows", "valid_core_rows", "registry_rows", "knn_k", "knn_enabled", "transition_lines_enabled",
        "transition_line_count", "registry_overlay_enabled", "plotly_available", "demo_mode",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in statuses:
            writer.writerow({
                "generated_at_utc": utc_now(),
                "record_type": "artifact",
                "artifact": s.label,
                "path": repo_rel(s.path),
                "required_for": s.required_for,
                "exists": s.exists,
                "status": s.status,
                "rows": "" if s.rows is None else s.rows,
                "columns": "" if s.columns is None else s.columns,
                "message": s.message,
                "feature_table_rows": state.metrics.feature_table_rows,
                "valid_core_rows": state.metrics.valid_core_rows,
                "registry_rows": state.metrics.registry_rows,
                "knn_k": state.metrics.knn_k,
                "knn_enabled": state.metrics.knn_enabled,
                "transition_lines_enabled": state.metrics.transition_lines_enabled,
                "transition_line_count": state.metrics.transition_line_count,
                "registry_overlay_enabled": state.metrics.registry_overlay_enabled,
                "plotly_available": state.metrics.plotly_available,
                "demo_mode": state.metrics.demo_mode,
            })
        for warning in state.warnings:
            writer.writerow({
                "generated_at_utc": utc_now(), "record_type": "warning", "artifact": "warning", "message": warning,
                "feature_table_rows": state.metrics.feature_table_rows,
                "valid_core_rows": state.metrics.valid_core_rows,
                "registry_rows": state.metrics.registry_rows,
                "knn_k": state.metrics.knn_k,
                "knn_enabled": state.metrics.knn_enabled,
                "transition_lines_enabled": state.metrics.transition_lines_enabled,
                "transition_line_count": state.metrics.transition_line_count,
                "registry_overlay_enabled": state.metrics.registry_overlay_enabled,
                "plotly_available": state.metrics.plotly_available,
                "demo_mode": state.metrics.demo_mode,
            })
    state.generated.append(manifest)
    return manifest


PLOT_HEIGHT_PX = 860


def plot_common_layout(fig: Any, title: str) -> None:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#05080b",
        plot_bgcolor="#05080b",
        autosize=True,
        height=PLOT_HEIGHT_PX,
        font={"family": "Menlo, Monaco, Consolas, monospace", "color": "#d8e3e7"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        legend={"bgcolor": "rgba(5,8,11,0.65)", "bordercolor": "#23303a", "borderwidth": 1},
        scene={
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "bgcolor": "#05080b",
            "xaxis": {"gridcolor": "#1b2a34", "zerolinecolor": "#37505f"},
            "yaxis": {"gridcolor": "#1b2a34", "zerolinecolor": "#37505f"},
            "zaxis": {"gridcolor": "#1b2a34", "zerolinecolor": "#37505f"},
        },
    )


def plotly_div(fig: Any) -> str:
    """Return a responsive Plotly div that actually fills the cockpit panel."""
    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        default_width="100%",
        default_height=f"{PLOT_HEIGHT_PX}px",
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )


def wrap_plotly_html(title: str, plot_div: str, side_html: str, demo: bool) -> str:
    body = f"""
<style>
.plot-grid {{
  display: grid;
  grid-template-columns: 330px minmax(0, 1fr);
  gap: 12px;
  padding: 12px;
  align-items: stretch;
}}
.plot-panel {{
  min-width: 0;
}}
.plot-host {{
  padding: 0;
  height: {PLOT_HEIGHT_PX}px;
  min-height: {PLOT_HEIGHT_PX}px;
  overflow: hidden;
}}
.plot-host .plotly-graph-div {{
  width: 100% !important;
  height: {PLOT_HEIGHT_PX}px !important;
}}
.plot-host .js-plotly-plot,
.plot-host .plot-container,
.plot-host .svg-container {{
  width: 100% !important;
  height: 100% !important;
}}
@media (max-width: 900px) {{
  .plot-grid {{ grid-template-columns: 1fr; }}
  .plot-host {{ height: 720px; min-height: 720px; }}
  .plot-host .plotly-graph-div {{ height: 720px !important; }}
}}
</style>
<div class="plot-grid">
  <section class="panel"><h2>Inspector</h2><div class="inner">{side_html}</div></section>
  <section class="panel plot-panel"><h2>{html.escape(title)}</h2><div class="inner plot-host">{plot_div}</div></section>
</div>
<div class="console">Hover points for relation/object details. Drag to rotate. Scroll to zoom. No fake surface interpolation is generated in v2.</div>
"""
    return html_shell(title, body, demo=demo)


def build_hover_columns(df: Any, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def add_transition_lines(fig: Any, go: Any, geom_df: Any, state: RunState) -> None:
    if geom_df is None or geom_df.empty:
        state.warnings.append("Transition lines skipped: no stability geometry rows available.")
        return
    if "object" not in geom_df.columns:
        state.warnings.append("Transition lines skipped: missing object / scale ordering columns.")
        return
    order_col = pick_column(geom_df.columns, ["scale_index_from", "scale_index_to", "scale_index", "candidate_rank"])
    if order_col is None:
        state.warnings.append("Transition lines skipped: missing object / scale ordering columns.")
        return
    required = STABILITY_CORE_COLUMNS + ["object", order_col]
    data = geom_df.dropna(subset=[c for c in required if c in geom_df.columns]).copy()
    if data.empty:
        state.warnings.append("Transition lines skipped: no valid rows after dropping missing object/scale/core coordinates.")
        return
    group_cols = ["object"]
    if "cohort" in data.columns:
        group_cols.append("cohort")
    count = 0
    for key, group in data.groupby(group_cols, dropna=True):
        group = group.sort_values(order_col)
        if len(group) < 2:
            continue
        name = " / ".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
        fig.add_trace(go.Scatter3d(
            x=group["mean_lambda_local_mean"],
            y=group["mean_delta_d_mean"],
            z=group["bounded_share_mean"],
            mode="lines",
            name=f"transition: {name}",
            line={"width": 2, "color": "rgba(160, 210, 255, 0.22)"},
            hoverinfo="skip",
            showlegend=False,
        ))
        count += 1
    if count == 0:
        state.warnings.append("Transition lines skipped: no groups with at least two ordered rows.")
        return
    state.metrics.transition_lines_enabled = True
    state.metrics.transition_line_count = count


def add_registry_overlay(fig: Any, registry_points: Any, geom_df: Any, state: RunState) -> None:
    if registry_points is None or registry_points.empty:
        state.warnings.append("Registry overlay skipped: registry artifact absent.")
        return
    if geom_df is None or geom_df.empty:
        state.warnings.append("Registry overlay skipped: stability geometry absent.")
        return
    valid = geom_df.dropna(subset=STABILITY_CORE_COLUMNS)
    if valid.empty:
        state.warnings.append("Registry overlay skipped: no valid stability-core bounds.")
        return
    x_min, x_max = float(valid["mean_lambda_local_mean"].min()), float(valid["mean_lambda_local_mean"].max())
    y_min, y_max = float(valid["mean_delta_d_mean"].min()), float(valid["mean_delta_d_mean"].max())
    z_min, z_max = float(valid["bounded_share_mean"].min()), float(valid["bounded_share_mean"].max())
    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0
    dz = (z_max - z_min) or 1.0
    anchors = registry_points.head(6).copy()
    if anchors.empty:
        state.warnings.append("Registry overlay skipped: no registry rows available.")
        return
    xs, ys, zs, labels = [], [], [], []
    for i, (_, row) in enumerate(anchors.iterrows()):
        xs.append(x_max + 0.08 * dx)
        ys.append(y_min + (i + 1) / (len(anchors) + 1) * dy)
        zs.append(z_max + 0.08 * dz)
        label = str(row.get("task", "relation"))
        status = str(row.get("rig_status", ""))
        labels.append(f"{label}<br>{status}")
    fig.add_trace({
        "type": "scatter3d",
        "x": xs,
        "y": ys,
        "z": zs,
        "mode": "markers+text",
        "name": "registry anchors",
        "text": labels,
        "textposition": "top center",
        "marker": {"size": 3, "color": "rgba(57,231,255,0.70)", "symbol": "diamond"},
        "hoverinfo": "text",
        "showlegend": True,
    })
    state.metrics.registry_overlay_enabled = True


def make_stability_core_view(outdir: Path, geom_df: Any, registry_points: Any, px: Any, go: Any, state: RunState, transition_lines: bool, registry_overlay: bool) -> Path | None:
    if geom_df is None or geom_df.empty:
        state.warnings.append("stability_core_3d.html skipped: stability geometry not available")
        return None
    missing = require_columns(geom_df, STABILITY_CORE_COLUMNS)
    if missing:
        state.warnings.append("stability_core_3d.html skipped: missing required columns " + ", ".join(missing))
        return None
    df = geom_df.copy()
    df = df.dropna(subset=STABILITY_CORE_COLUMNS)
    if df.empty:
        state.warnings.append("stability_core_3d.html skipped: no numeric rows for stability-core axes")
        return None

    color_col = "case" if "case" in df.columns else "bounded_stability_signature"
    hover_cols = build_hover_columns(df, [
        "case", "object", "cohort", "scale_index_from", "scale_index_to",
        "divergence_mean", "unboundedness", "instability_signature", "bounded_stability_signature",
        "local_density", "local_case_purity", "local_boundary_score", "distance_to_case_centroid", "distance_to_global_centroid",
        "dominant_family", "dominant_reason", "n_paths", "demo_notice",
    ])
    fig = px.scatter_3d(
        df,
        x="mean_lambda_local_mean",
        y="mean_delta_d_mean",
        z="bounded_share_mean",
        color=color_col,
        color_discrete_map=CASE_COLORS if color_col == "case" else None,
        hover_data=hover_cols,
        opacity=0.86,
        title="Stability Core Geometry — raw core plus derived local structure",
    )
    fig.update_traces(marker={"size": 4, "line": {"width": 0}})
    if transition_lines:
        add_transition_lines(fig, go, df, state)
    else:
        state.warnings.append("Transition lines disabled by --no-transition-lines.")
    if registry_overlay:
        add_registry_overlay(fig, registry_points, df, state)
    else:
        state.warnings.append("Registry overlay disabled by --no-registry-overlay.")
    plot_common_layout(fig, "Stability Core Geometry")
    fig.update_layout(scene={
        "xaxis_title": "mean_lambda_local_mean",
        "yaxis_title": "mean_delta_d_mean",
        "zaxis_title": "bounded_share_mean",
        "bgcolor": "#05080b",
    })
    plot_div = plotly_div(fig)
    side = f"""
<p><span class='badge good'>valid core rows</span> {len(df)}</p>
<p><span class='badge'>x</span> mean_lambda_local_mean</p>
<p><span class='badge'>y</span> mean_delta_d_mean</p>
<p><span class='badge'>z</span> bounded_share_mean</p>
<p><span class='badge'>color</span> {html.escape(str(color_col))}</p>
<h3>Derived fields</h3>
<p>divergence_mean<br>unboundedness<br>instability_signature<br>bounded_stability_signature</p>
<h3>Local geometry</h3>
<p>kNN enabled: {state.metrics.knn_enabled}<br>k: {state.metrics.knn_k}<br>transition traces: {state.metrics.transition_line_count}<br>registry overlay: {state.metrics.registry_overlay_enabled}</p>
<p class='muted'>Source: OBS-078a feature table. Registry labels are secondary anchors only; they are not treated as real stability-core coordinates.</p>
"""
    path = outdir / "stability_core_3d.html"
    path.write_text(wrap_plotly_html("Stability Core Geometry", plot_div, side, state.demo), encoding="utf-8")
    state.generated.append(path)
    return path


def make_registry_view(outdir: Path, registry_points: Any, px: Any, state: RunState) -> Path | None:
    if registry_points is None or registry_points.empty:
        state.warnings.append("rig_registry_3d.html skipped: registry plot points not available")
        return None
    required = ["x_task_index", "y_carrier_index", "z_survival", "rig_status"]
    missing = require_columns(registry_points, required)
    if missing:
        state.warnings.append("rig_registry_3d.html skipped: missing required columns " + ", ".join(missing))
        return None
    df = registry_points.copy()
    df["z_survival"] = pd.to_numeric(df["z_survival"], errors="coerce")
    df = df.dropna(subset=["z_survival"])
    if df.empty:
        state.warnings.append("rig_registry_3d.html skipped: no numeric z_survival rows")
        return None
    hover_cols = build_hover_columns(df, [
        "relation_id", "task", "carrier", "carrier_role", "rig_status",
        "task_geometry_needed_level", "repair_recommendation", "obs080c_carrier_ba",
        "obs080d_carrier_mean_ba", "obs080d_carrier_min_ci95_low", "demo_notice",
    ])
    symbol_col = "carrier_role" if "carrier_role" in df.columns else None
    size_col = pick_column(df.columns, ["obs080c_carrier_ba", "mean_survival_score", "n_survival_rows", "support"])
    fig = px.scatter_3d(
        df,
        x="x_task_index",
        y="y_carrier_index",
        z="z_survival",
        color="rig_status",
        symbol=symbol_col,
        size=size_col,
        color_discrete_map=STATUS_COLORS,
        hover_data=hover_cols,
        opacity=0.92,
        title="RIG Registry 3D — task × carrier role × survival",
    )
    fig.update_traces(marker={"line": {"width": 0}})
    task_ticks = df[["x_task_index", "task"]].drop_duplicates().sort_values("x_task_index")
    carrier_ticks = df[["y_carrier_index", "carrier_role"]].drop_duplicates().sort_values("y_carrier_index")
    plot_common_layout(fig, "RIG Registry 3D")
    fig.update_layout(scene={
        "xaxis_title": "task",
        "yaxis_title": "carrier_role",
        "zaxis_title": "z_survival",
        "xaxis": {"tickmode": "array", "tickvals": task_ticks["x_task_index"].tolist(), "ticktext": task_ticks["task"].tolist(), "gridcolor": "#1b2a34"},
        "yaxis": {"tickmode": "array", "tickvals": carrier_ticks["y_carrier_index"].tolist(), "ticktext": carrier_ticks["carrier_role"].tolist(), "gridcolor": "#1b2a34"},
        "zaxis": {"gridcolor": "#1b2a34"},
        "bgcolor": "#05080b",
    })
    plot_div = plotly_div(fig)
    status_counts = df["rig_status"].value_counts(dropna=False).to_dict()
    counts_html = "".join(f"<p><span class='badge'>{html.escape(str(k))}</span> {v}</p>" for k, v in status_counts.items())
    side = f"""
<p><span class='badge good'>registry rows</span> {len(df)}</p>
<p><span class='badge'>x</span> x_task_index</p>
<p><span class='badge'>y</span> y_carrier_index</p>
<p><span class='badge'>z</span> z_survival</p>
<h3>Status counts</h3>{counts_html}
<p class='muted'>Source: derived rig_registry_plot_points.csv from OBS-081 relation registry.</p>
"""
    path = outdir / "rig_registry_3d.html"
    path.write_text(wrap_plotly_html("RIG Registry 3D", plot_div, side, state.demo), encoding="utf-8")
    state.generated.append(path)
    return path


def make_index(outdir: Path, statuses: list[ArtifactStatus], state: RunState, registry_points: Any | None, geom_df: Any | None) -> Path:
    link_names = [
        "stability_core_3d.html", "rig_registry_3d.html", "rig_stability_core_geometry.csv",
        "rig_registry_plot_points.csv", "rig_navigator_manifest.csv",
    ]
    links = []
    for name in link_names:
        p = outdir / name
        if p.exists():
            links.append(f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>")
        else:
            links.append(f"<li><span class='muted'>{html.escape(name)} not generated</span></li>")
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in state.warnings) or "<li>None</li>"
    errors = "".join(f"<li>{html.escape(e)}</li>" for e in state.errors) or "<li>None</li>"
    plotly_msg = "available" if state.metrics.plotly_available else state.plotly_hint
    metrics = state.metrics
    body = f"""
<div class="grid">
  <section class="panel"><h2>Artifact Status</h2><div class="inner">{status_table(statuses)}</div></section>
  <section class="panel"><h2>Navigator Outputs</h2><div class="inner"><ul>{''.join(links)}</ul><p class="muted">Plotly: {html.escape(plotly_msg)}</p></div></section>
  <section class="panel"><h2>v2 Geometry Metrics</h2><div class="inner"><pre>feature_table_rows: {metrics.feature_table_rows}
valid_core_rows: {metrics.valid_core_rows}
registry_rows: {metrics.registry_rows}
knn_k: {metrics.knn_k}
knn_enabled: {metrics.knn_enabled}
transition_lines_enabled: {metrics.transition_lines_enabled}
transition_line_count: {metrics.transition_line_count}
registry_overlay_enabled: {metrics.registry_overlay_enabled}
plotly_available: {metrics.plotly_available}
demo_mode: {metrics.demo_mode}</pre></div></section>
  <section class="panel"><h2>Stability Geometry Preview</h2><div class="inner">{dataframe_preview(geom_df)}</div></section>
  <section class="panel"><h2>Registry Plot Points Preview</h2><div class="inner">{dataframe_preview(registry_points)}</div></section>
</div>
<div class="console">
  <div><span class="green">Scope:</span> local RIG Navigator v2 derived geometry instrument. Generated outputs under outputs/ are file-first and may be gitignored.</div>
  <div><span class="orange">Missing artifacts:</span> expected under outputs/ may be absent from public checkout; run OBS scripts or pass explicit paths.</div>
  <div><span class="cyan">Warnings:</span><ul>{warnings}</ul></div>
  <div><span class="cyan">Errors:</span><ul>{errors}</ul></div>
</div>
"""
    index = outdir / "rig_navigator_index.html"
    index.write_text(html_shell("PAM Observatory — RIG Navigator v2", body, demo=state.demo), encoding="utf-8")
    state.generated.append(index)
    return index


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = RunState(demo=args.demo)
    state.metrics.demo_mode = args.demo
    state.metrics.knn_k = args.knn_k
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    registry_df, registry_status = load_csv(args.registry, "rig_relation_registry", "rig_registry_3d / registry anchors", state)
    survival_df, survival_status = load_csv(args.survival, "rig_survival_matrix", "index/diagnostics", state)
    feature_df, feature_status = load_csv(args.feature_table, "obs078a_feature_table", "stability core geometry", state)
    statuses = [registry_status, survival_status, feature_status]

    if args.demo:
        if pd is None:
            state.errors.append(f"--demo requires pandas, but pandas import failed: {PANDAS_IMPORT_ERROR}")
        else:
            if registry_df is None:
                registry_df = make_demo_registry()
                registry_status.status = "demo"
                registry_status.rows = len(registry_df)
                registry_status.columns = len(registry_df.columns)
                registry_status.message = DEMO_BANNER
            if survival_df is None and registry_df is not None:
                survival_df = registry_df[["relation_id", "task", "carrier_role", "mean_survival_score"]].copy()
                survival_status.status = "demo"
                survival_status.rows = len(survival_df)
                survival_status.columns = len(survival_df.columns)
                survival_status.message = DEMO_BANNER
            if feature_df is None:
                feature_df = make_demo_feature_table()
                feature_status.status = "demo"
                feature_status.rows = len(feature_df)
                feature_status.columns = len(feature_df.columns)
                feature_status.message = DEMO_BANNER
    else:
        for s in statuses:
            if s.status == "missing":
                state.warnings.append(
                    f"{s.label} missing at {repo_rel(s.path)}; generated outputs under outputs/ may be gitignored. Pass explicit paths or rerun with --demo for mock data."
                )

    geom_df = compute_stability_geometry(feature_df, outdir, state, args.knn_k)
    registry_points = compute_registry_plot_points(registry_df, outdir, state)

    px, go, plotly_error = import_plotly()
    state.metrics.plotly_available = px is not None and go is not None
    if not state.metrics.plotly_available:
        state.warnings.append("Plotly missing: wrote data products and index only. " + state.plotly_hint + (f" ({plotly_error})" if plotly_error else ""))
    else:
        make_stability_core_view(
            outdir,
            geom_df,
            registry_points,
            px,
            go,
            state,
            transition_lines=not args.no_transition_lines,
            registry_overlay=not args.no_registry_overlay,
        )
        make_registry_view(outdir, registry_points, px, state)

    manifest = write_manifest(outdir, statuses, state)
    index = make_index(outdir, statuses, state, registry_points, geom_df)

    print("RIG Navigator v2 generation complete")
    print(f"  index:    {repo_rel(index)}")
    print(f"  manifest: {repo_rel(manifest)}")
    for path in state.generated:
        if path not in {index, manifest}:
            print(f"  output:   {repo_rel(path)}")
    if state.warnings:
        print("\nWarnings:")
        for warning in state.warnings:
            print(f"  - {warning}")
    if state.errors:
        print("\nErrors:")
        for error in state.errors:
            print(f"  - {error}")

    if args.open_index:
        try:
            webbrowser.open(index.resolve().as_uri())
        except Exception as exc:
            print(f"Could not open browser: {exc}", file=sys.stderr)

    return 1 if state.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
