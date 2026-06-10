#!/usr/bin/env python3
"""
figure_obs076_scale_space_stack.py

OBS-076 figure utility — Scale-space stack visualization

v2 patch
--------
v2 improves the interactive HTML mode and adds object-tube visualization.

New options:
    --only-object
        Plot only nodes belonging to the selected OBS-076c object.

    --no-background-cloud
        Hide the background cloud in static and interactive views.

    --html-node-traces
        In HTML mode, add node trajectory traces as a separate toggleable layer.

    --html-scale-slices
        In HTML mode, add scale-slice bounding boxes as separate toggleable layers.

    --object-color-by field|scale|membership
        Controls coloring of highlighted object nodes.

    --object-alpha
        Separate alpha for highlighted object nodes in static PNG mode.

Purpose
-------
Visualize the OBS-076 scale-space structure as a 3D stack:

    x = OBS-076b rebuilt observable-space coordinate geom_x
    y = OBS-076b rebuilt observable-space coordinate geom_y
    z = diffusion scale, usually log10(t)
    color = selected field

Optional object membership from OBS-076c can be overlaid as highlighted nodes.

This is intended to visualize:

    - support migration
    - coarse/fine scale organization
    - factorization across scale
    - object tubes across diffusion scale

Scope
-----
This is a visualization helper.

It does not compute new scale-space results.

It consumes existing OBS-076a/076b/076c artifacts.

Inputs
------
Required:
    --node-geometry obs076b_node_geometry_by_scale.csv

Optional:
    --bundle obs076a_diffusion_bundle.npz
        Injects dynamic observable fields as dyn__<observable>.

    --objects obs076c_object_membership_by_scale.csv
        Allows object membership overlay, e.g. energy_ridge.

Outputs
-------
Default output:
    <outdir>/<case>_scale_space_stack_<field>.png

Optional:
    --html
        Writes an interactive Plotly HTML if plotly is available.

Examples
--------
Cp3 energy stack, interactive:

    PYTHONPATH=src .venv/bin/python experiments/figures/figure_obs076_scale_space_stack.py \\
      --case Cp3 \\
      --node-geometry outputs/corpora/Cp3/campaigns/full_v1/pipeline/obs076b_geometry_rebuild_shared14_mds_pilot/obs076b_node_geometry_by_scale.csv \\
      --bundle outputs/corpora/Cp3/campaigns/full_v1/pipeline/obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_no_detcond/obs076a_diffusion_bundle.npz \\
      --objects outputs/corpora/Cp3/campaigns/full_v1/pipeline/obs076c_structural_object_persistence_shared14_mds_pilot_v2/obs076c_object_membership_by_scale.csv \\
      --field energy \\
      --object energy_ridge \\
      --html \\
      --html-node-traces \\
      --html-scale-slices \\
      --outdir outputs/figures/obs076_scale_space_stack/interactive

Cp3 object tube only:

    PYTHONPATH=src .venv/bin/python experiments/figures/figure_obs076_scale_space_stack.py \\
      --case Cp3 \\
      --node-geometry outputs/corpora/Cp3/campaigns/full_v1/pipeline/obs076b_geometry_rebuild_shared14_mds_pilot/obs076b_node_geometry_by_scale.csv \\
      --bundle outputs/corpora/Cp3/campaigns/full_v1/pipeline/obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_no_detcond/obs076a_diffusion_bundle.npz \\
      --objects outputs/corpora/Cp3/campaigns/full_v1/pipeline/obs076c_structural_object_persistence_shared14_mds_pilot_v2/obs076c_object_membership_by_scale.csv \\
      --field dyn__signed_phase \\
      --object phase_band_positive \\
      --only-object \\
      --html \\
      --html-node-traces \\
      --outdir outputs/figures/obs076_scale_space_stack/object_tubes
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DYN_PREFIX = "dyn__"
OBJ_PREFIX = "obj__"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create 3D OBS-076 scale-space stack visualizations."
    )

    parser.add_argument("--case", default="case")
    parser.add_argument("--node-geometry", required=True, type=Path)
    parser.add_argument("--bundle", default=None, type=Path)
    parser.add_argument("--objects", default=None, type=Path)
    parser.add_argument("--id-col", default="id")

    parser.add_argument(
        "--field",
        default="energy",
        help=(
            "Field to color by. Can be an OBS-076b field like energy, "
            "or an injected OBS-076a field like dyn__signed_phase."
        ),
    )
    parser.add_argument(
        "--object",
        default=None,
        help=(
            "Optional OBS-076c object to highlight, e.g. energy_ridge, "
            "response_ridge, phase_band_positive."
        ),
    )

    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--filename", default=None)

    parser.add_argument(
        "--z-mode",
        choices=["log10_t", "t", "scale_index"],
        default="log10_t",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--cmap", default="viridis")

    parser.add_argument("--view-elev", type=float, default=24)
    parser.add_argument("--view-azim", type=float, default=-58)
    parser.add_argument("--point-size", type=float, default=22)
    parser.add_argument("--highlight-size", type=float, default=72)
    parser.add_argument("--alpha", type=float, default=0.86)
    parser.add_argument("--object-alpha", type=float, default=0.96)

    parser.add_argument("--draw-scale-slices", action="store_true")
    parser.add_argument("--draw-node-traces", action="store_true")
    parser.add_argument(
        "--trace-alpha",
        type=float,
        default=0.16,
        help="Alpha for node traces when --draw-node-traces is enabled.",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=75,
        help="Maximum number of node traces to draw.",
    )

    parser.add_argument(
        "--only-object",
        action="store_true",
        help="Plot only rows belonging to --object. Requires --object and --objects.",
    )
    parser.add_argument(
        "--no-background-cloud",
        action="store_true",
        help="Do not draw non-highlighted/background nodes.",
    )
    parser.add_argument(
        "--object-color-by",
        choices=["field", "scale", "membership"],
        default="field",
        help="How to color highlighted object nodes.",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional color minimum.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional color maximum.",
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write an interactive Plotly HTML if plotly is installed.",
    )
    parser.add_argument(
        "--html-node-traces",
        action="store_true",
        help="In HTML mode, add toggleable node trajectory traces.",
    )
    parser.add_argument(
        "--html-scale-slices",
        action="store_true",
        help="In HTML mode, add toggleable scale slice boxes.",
    )
    parser.add_argument(
        "--html-trace-limit",
        type=int,
        default=None,
        help="Maximum node traces in HTML. Defaults to --trace-limit.",
    )
    parser.add_argument(
        "--html-marker-size",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--html-object-marker-size",
        type=float,
        default=7.5,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
    )

    args = parser.parse_args()

    if args.only_object and not args.object:
        raise ValueError("--only-object requires --object")

    if args.object_color_by in {"scale", "membership"} and not args.object:
        raise ValueError("--object-color-by scale|membership requires --object")

    return args


def as_str_array(x: np.ndarray) -> np.ndarray:
    return np.array([str(v) for v in x.tolist()], dtype=str)


def slugify(x: str) -> str:
    x = x.strip()
    x = x.replace(DYN_PREFIX, "dyn_")
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def load_node_geometry(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [id_col, "scale_index", "t", "geom_x", "geom_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"node geometry missing required columns: {missing}")

    df[id_col] = df[id_col].astype(str)
    df["scale_index"] = pd.to_numeric(df["scale_index"], errors="raise").astype(int)
    df["t"] = pd.to_numeric(df["t"], errors="raise").astype(float)
    df["geom_x"] = pd.to_numeric(df["geom_x"], errors="coerce")
    df["geom_y"] = pd.to_numeric(df["geom_y"], errors="coerce")

    if df.duplicated([id_col, "scale_index"]).any():
        dup = df[df.duplicated([id_col, "scale_index"])][[id_col, "scale_index"]].head(10)
        raise ValueError(f"duplicate node/scale rows:\n{dup}")

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

    return {
        "ids": ids,
        "observable_cols": observable_cols,
        "X_t": x_t,
        "t": ts,
    }


def inject_bundle_features(
    df: pd.DataFrame,
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
            f"OBS-076b scale count {len(scale_values)} does not match bundle scale count {len(ts)}"
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

    merged = df.merge(
        dyn,
        on=[id_col, "scale_index"],
        how="left",
        validate="one_to_one",
    )

    return merged, dyn_cols, "ok"


def load_object_memberships(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    obj = pd.read_csv(path)

    required = [id_col, "scale_index", "object"]
    missing = [c for c in required if c not in obj.columns]
    if missing:
        raise ValueError(f"object membership missing required columns: {missing}")

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
    objects_path: Path | None,
    id_col: str,
) -> tuple[pd.DataFrame, list[str], str]:
    if objects_path is None:
        return df, [], "not_provided"

    wide = load_object_memberships(objects_path, id_col)
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


def add_z_column(df: pd.DataFrame, z_mode: str) -> pd.DataFrame:
    df = df.copy()

    if z_mode == "scale_index":
        df["z_scale"] = df["scale_index"].astype(float)
        z_label = "scale index"
    elif z_mode == "t":
        df["z_scale"] = df["t"].astype(float)
        z_label = "diffusion scale t"
    elif z_mode == "log10_t":
        t = pd.to_numeric(df["t"], errors="coerce")
        df["z_scale"] = np.log10(np.maximum(t, 1e-12))
        z_label = "log10 diffusion scale"
    else:
        raise ValueError(f"unknown z_mode: {z_mode}")

    df.attrs["z_label"] = z_label
    return df


def robust_color_limits(values: pd.Series, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0

    lo = float(np.nanpercentile(x, 2)) if vmin is None else float(vmin)
    hi = float(np.nanpercentile(x, 98)) if vmax is None else float(vmax)

    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if abs(hi - lo) < 1e-12:
            hi = lo + 1.0

    return lo, hi


def safe_field_label(field: str) -> str:
    return field.replace(DYN_PREFIX, "").replace("_", " ")


def default_filename(case: str, field: str, obj: str | None, only_object: bool) -> str:
    clean_field = slugify(field)
    prefix = f"{slugify(case)}_scale_space_stack_{clean_field}"

    if obj:
        prefix += f"_highlight_{slugify(obj)}"
    if only_object:
        prefix += "_only_object"

    return f"{prefix}.png"


def object_column(obj: str | None) -> str | None:
    if obj is None:
        return None
    return f"{OBJ_PREFIX}{obj}"


def finite_base_mask(df: pd.DataFrame, field: str) -> pd.Series:
    return (
        np.isfinite(pd.to_numeric(df["geom_x"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["geom_y"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["z_scale"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df[field], errors="coerce"))
    )


def prepare_plot_df(
    df: pd.DataFrame,
    field: str,
    obj: str | None,
    only_object: bool,
) -> pd.DataFrame:
    if field not in df.columns:
        available = ", ".join(sorted(df.columns))
        raise ValueError(f"field {field!r} not in table. Available columns:\n{available}")

    work = df.copy()
    work[field] = pd.to_numeric(work[field], errors="coerce")

    obj_col = object_column(obj)
    if obj_col is not None and obj_col not in work.columns:
        available_objs = sorted(c.replace(OBJ_PREFIX, "") for c in work.columns if c.startswith(OBJ_PREFIX))
        raise ValueError(f"object {obj!r} not found. Available objects: {available_objs}")

    valid = work[finite_base_mask(work, field)].copy()

    if only_object:
        if obj_col is None:
            raise ValueError("--only-object requires --object")
        valid = valid[pd.to_numeric(valid[obj_col], errors="coerce").fillna(0) > 0].copy()

    if valid.empty:
        raise ValueError(f"No finite rows available for field {field}")

    return valid


def object_color_values(
    df: pd.DataFrame,
    field: str,
    obj: str | None,
    object_color_by: str,
) -> tuple[np.ndarray, str, str | None]:
    if object_color_by == "field":
        return pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=float), safe_field_label(field), "Viridis"

    if object_color_by == "scale":
        return pd.to_numeric(df["z_scale"], errors="coerce").to_numpy(dtype=float), df.attrs.get("z_label", "scale"), "Viridis"

    if object_color_by == "membership":
        if obj is None:
            raise ValueError("membership coloring requires --object")
        return np.ones(len(df), dtype=float), f"{obj} membership", None

    raise ValueError(f"unknown object_color_by: {object_color_by}")


def plot_matplotlib_stack(
    df: pd.DataFrame,
    case: str,
    field: str,
    obj: str | None,
    outpath: Path,
    cmap: str,
    view_elev: float,
    view_azim: float,
    point_size: float,
    highlight_size: float,
    alpha: float,
    object_alpha: float,
    draw_scale_slices: bool,
    draw_node_traces: bool,
    trace_alpha: float,
    trace_limit: int,
    vmin: float | None,
    vmax: float | None,
    title: str | None,
    dpi: int,
    id_col: str,
    only_object: bool,
    no_background_cloud: bool,
    object_color_by: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    valid = prepare_plot_df(df, field=field, obj=obj, only_object=only_object)
    obj_col = object_column(obj)

    cmin, cmax = robust_color_limits(valid[field], vmin, vmax)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if draw_node_traces:
        trace_df = valid if only_object else df[finite_base_mask(df, field)].copy()
        if only_object and obj_col is not None:
            trace_df = trace_df[pd.to_numeric(trace_df[obj_col], errors="coerce").fillna(0) > 0]

        ids = trace_df[id_col].drop_duplicates().astype(str).head(trace_limit)
        for node_id in ids:
            sub = trace_df[trace_df[id_col].astype(str) == node_id].sort_values("scale_index")
            if len(sub) < 2:
                continue
            ax.plot(
                sub["geom_x"],
                sub["geom_y"],
                sub["z_scale"],
                linewidth=0.7,
                alpha=trace_alpha,
                color="black",
                zorder=1,
            )

    if draw_scale_slices:
        slice_df = valid if only_object else df[finite_base_mask(df, field)].copy()
        for scale_index, sub in slice_df.groupby("scale_index", sort=True):
            z = float(sub["z_scale"].iloc[0])
            x_min, x_max = float(sub["geom_x"].min()), float(sub["geom_x"].max())
            y_min, y_max = float(sub["geom_y"].min()), float(sub["geom_y"].max())
            ax.plot(
                [x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                [z, z, z, z, z],
                color="0.82",
                linewidth=0.6,
                alpha=0.45,
                zorder=0,
            )

    sc = None

    if not no_background_cloud and not only_object:
        sc = ax.scatter(
            valid["geom_x"],
            valid["geom_y"],
            valid["z_scale"],
            c=valid[field],
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            vmin=cmin,
            vmax=cmax,
            depthshade=True,
            linewidths=0,
            zorder=2,
        )

    legend_handles = []

    if obj:
        if only_object:
            h = valid.copy()
        else:
            h = valid[pd.to_numeric(valid[obj_col], errors="coerce").fillna(0) > 0].copy()

        if not h.empty:
            if object_color_by == "membership":
                obj_sc = ax.scatter(
                    h["geom_x"],
                    h["geom_y"],
                    h["z_scale"],
                    s=highlight_size,
                    facecolors="white",
                    edgecolors="black",
                    linewidths=1.1,
                    alpha=object_alpha,
                    depthshade=False,
                    zorder=4,
                )
            elif object_color_by == "scale":
                obj_sc = ax.scatter(
                    h["geom_x"],
                    h["geom_y"],
                    h["z_scale"],
                    c=h["z_scale"],
                    cmap=cmap,
                    s=highlight_size,
                    edgecolors="black",
                    linewidths=1.0,
                    alpha=object_alpha,
                    depthshade=False,
                    zorder=4,
                )
                sc = obj_sc if sc is None else sc
            else:
                obj_sc = ax.scatter(
                    h["geom_x"],
                    h["geom_y"],
                    h["z_scale"],
                    c=h[field],
                    cmap=cmap,
                    s=highlight_size,
                    edgecolors="black",
                    linewidths=1.0,
                    alpha=object_alpha,
                    vmin=cmin,
                    vmax=cmax,
                    depthshade=False,
                    zorder=4,
                )
                sc = obj_sc if sc is None else sc

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    label=f"{obj} membership",
                    markerfacecolor="none",
                    markersize=7,
                    linewidth=0,
                )
            )

    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
        if obj and object_color_by == "scale" and (only_object or no_background_cloud):
            cb.set_label(df.attrs.get("z_label", "scale"), rotation=270, labelpad=18)
        else:
            cb.set_label(safe_field_label(field), rotation=270, labelpad=18)

    ax.set_xlabel("geom_x")
    ax.set_ylabel("geom_y")
    ax.set_zlabel(df.attrs.get("z_label", "scale"))

    if title is None:
        title = f"{case} scale-space stack — {safe_field_label(field)}"
        if obj:
            title += f" / {obj}"
        if only_object:
            title += " only"

    ax.set_title(title, pad=20)

    ax.view_init(elev=view_elev, azim=view_azim)

    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)
    ax.grid(True, alpha=0.25)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_plotly_scale_slices(fig, df: pd.DataFrame, name_prefix: str = "scale slice") -> None:
    import plotly.graph_objects as go

    for scale_index, sub in df.groupby("scale_index", sort=True):
        if sub.empty:
            continue

        z = float(sub["z_scale"].iloc[0])
        x_min, x_max = float(sub["geom_x"].min()), float(sub["geom_x"].max())
        y_min, y_max = float(sub["geom_y"].min()), float(sub["geom_y"].max())

        fig.add_trace(
            go.Scatter3d(
                x=[x_min, x_max, x_max, x_min, x_min],
                y=[y_min, y_min, y_max, y_max, y_min],
                z=[z, z, z, z, z],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.28)", width=2),
                name=f"{name_prefix} {int(scale_index)}",
                legendgroup="scale_slices",
                visible="legendonly",
                hoverinfo="skip",
            )
        )


def add_plotly_node_traces(
    fig,
    df: pd.DataFrame,
    id_col: str,
    trace_limit: int,
    name_prefix: str = "node trace",
) -> None:
    import plotly.graph_objects as go

    ids = df[id_col].drop_duplicates().astype(str).head(trace_limit)

    first = True
    for node_id in ids:
        sub = df[df[id_col].astype(str) == node_id].sort_values("scale_index")
        if len(sub) < 2:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=sub["geom_x"],
                y=sub["geom_y"],
                z=sub["z_scale"],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.25)", width=2),
                name=name_prefix if first else name_prefix,
                legendgroup="node_traces",
                showlegend=first,
                visible="legendonly",
                hoverinfo="skip",
            )
        )
        first = False


def write_plotly_html(
    df: pd.DataFrame,
    case: str,
    field: str,
    obj: str | None,
    outpath: Path,
    title: str | None,
    id_col: str,
    only_object: bool,
    no_background_cloud: bool,
    object_color_by: str,
    html_node_traces: bool,
    html_scale_slices: bool,
    html_trace_limit: int,
    html_marker_size: float,
    html_object_marker_size: float,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError("Plotly is not installed; cannot write HTML") from exc

    valid = prepare_plot_df(df, field=field, obj=obj, only_object=only_object)
    full_valid = df[finite_base_mask(df, field)].copy()

    obj_col = object_column(obj)

    if title is None:
        title = f"{case} scale-space stack — {safe_field_label(field)}"
        if obj:
            title += f" / {obj}"
        if only_object:
            title += " only"

    fig = go.Figure()

    if not no_background_cloud and not only_object:
        fig.add_trace(
            go.Scatter3d(
                x=valid["geom_x"],
                y=valid["geom_y"],
                z=valid["z_scale"],
                mode="markers",
                marker=dict(
                    size=html_marker_size,
                    color=valid[field],
                    colorscale="Viridis",
                    opacity=0.78,
                    colorbar=dict(title=safe_field_label(field)),
                ),
                customdata=np.stack(
                    [
                        valid[id_col].astype(str).to_numpy(),
                        valid["scale_index"].to_numpy(),
                        valid["t"].to_numpy(),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "background node<br>"
                    "id=%{customdata[0]}<br>"
                    "scale_index=%{customdata[1]}<br>"
                    "t=%{customdata[2]:.6g}<br>"
                    f"{field}=%{{marker.color:.6g}}"
                    "<extra></extra>"
                ),
                name="background cloud",
                legendgroup="background",
            )
        )

    if obj:
        if only_object:
            h = valid.copy()
        else:
            h = valid[pd.to_numeric(valid[obj_col], errors="coerce").fillna(0) > 0].copy()

        if not h.empty:
            color_values, color_title, colorscale = object_color_values(
                h,
                field=field,
                obj=obj,
                object_color_by=object_color_by,
            )

            marker = dict(
                size=html_object_marker_size,
                opacity=0.96,
                line=dict(color="black", width=4),
            )

            if object_color_by == "membership":
                marker["color"] = "white"
            else:
                marker["color"] = color_values
                marker["colorscale"] = colorscale or "Viridis"
                if no_background_cloud or only_object:
                    marker["colorbar"] = dict(title=color_title)

            fig.add_trace(
                go.Scatter3d(
                    x=h["geom_x"],
                    y=h["geom_y"],
                    z=h["z_scale"],
                    mode="markers",
                    marker=marker,
                    customdata=np.stack(
                        [
                            h[id_col].astype(str).to_numpy(),
                            h["scale_index"].to_numpy(),
                            h["t"].to_numpy(),
                            pd.to_numeric(h[field], errors="coerce").to_numpy(dtype=float),
                        ],
                        axis=1,
                    ),
                    hovertemplate=(
                        f"{obj} member<br>"
                        "id=%{customdata[0]}<br>"
                        "scale_index=%{customdata[1]}<br>"
                        "t=%{customdata[2]:.6g}<br>"
                        f"{field}=%{{customdata[3]:.6g}}"
                        "<extra></extra>"
                    ),
                    name=f"{obj} membership",
                    legendgroup="object",
                )
            )

    if html_node_traces:
        trace_df = valid if only_object else full_valid
        if only_object and obj_col is not None:
            trace_df = trace_df[pd.to_numeric(trace_df[obj_col], errors="coerce").fillna(0) > 0]
        add_plotly_node_traces(
            fig=fig,
            df=trace_df,
            id_col=id_col,
            trace_limit=html_trace_limit,
        )

    if html_scale_slices:
        slice_df = valid if only_object else full_valid
        add_plotly_scale_slices(fig=fig, df=slice_df)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="geom_x",
            yaxis_title="geom_y",
            zaxis_title=df.attrs.get("z_label", "scale"),
            xaxis=dict(showbackground=True, backgroundcolor="rgba(248,248,248,0.65)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgba(248,248,248,0.65)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgba(248,248,248,0.65)"),
        ),
        legend=dict(
            title="Layers",
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.write_html(outpath)


def write_manifest(
    args,
    outdir: Path,
    png_path: Path,
    html_path: Path | None,
    dyn_cols: list[str],
    obj_cols: list[str],
    dyn_status: str,
    obj_status: str,
    n_rows_total: int,
    n_rows_plotted: int,
) -> None:
    rows = [
        {
            "artifact": "node_geometry",
            "path": str(args.node_geometry),
            "status": "ok",
            "details": "",
        },
        {
            "artifact": "bundle",
            "path": str(args.bundle) if args.bundle else "",
            "status": dyn_status,
            "details": f"dynamic_columns={len(dyn_cols)}",
        },
        {
            "artifact": "objects",
            "path": str(args.objects) if args.objects else "",
            "status": obj_status,
            "details": f"object_columns={len(obj_cols)}",
        },
        {
            "artifact": "png",
            "path": str(png_path),
            "status": "written",
            "details": "",
        },
        {
            "artifact": "field",
            "path": "",
            "status": "configured",
            "details": args.field,
        },
        {
            "artifact": "object",
            "path": "",
            "status": "configured" if args.object else "not_configured",
            "details": args.object or "",
        },
        {
            "artifact": "plot_config",
            "path": "",
            "status": "configured",
            "details": json.dumps(
                {
                    "z_mode": args.z_mode,
                    "only_object": bool(args.only_object),
                    "no_background_cloud": bool(args.no_background_cloud),
                    "object_color_by": args.object_color_by,
                    "draw_node_traces": bool(args.draw_node_traces),
                    "draw_scale_slices": bool(args.draw_scale_slices),
                    "html": bool(args.html),
                    "html_node_traces": bool(args.html_node_traces),
                    "html_scale_slices": bool(args.html_scale_slices),
                    "n_rows_total": int(n_rows_total),
                    "n_rows_plotted": int(n_rows_plotted),
                },
                sort_keys=True,
            ),
        },
    ]

    if html_path is not None:
        rows.append(
            {
                "artifact": "html",
                "path": str(html_path),
                "status": "written",
                "details": "",
            }
        )

    pd.DataFrame(rows).to_csv(outdir / "figure_obs076_scale_space_stack_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_node_geometry(args.node_geometry, args.id_col)

    df, dyn_cols, dyn_status = inject_bundle_features(
        df=df,
        bundle_path=args.bundle,
        id_col=args.id_col,
    )

    df, obj_cols, obj_status = inject_object_features(
        df=df,
        objects_path=args.objects,
        id_col=args.id_col,
    )

    df = add_z_column(df, args.z_mode)

    plotted_df = prepare_plot_df(
        df,
        field=args.field,
        obj=args.object,
        only_object=args.only_object,
    )

    filename = args.filename or default_filename(
        args.case,
        args.field,
        args.object,
        args.only_object,
    )
    png_path = args.outdir / filename

    plot_matplotlib_stack(
        df=df,
        case=args.case,
        field=args.field,
        obj=args.object,
        outpath=png_path,
        cmap=args.cmap,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        point_size=args.point_size,
        highlight_size=args.highlight_size,
        alpha=args.alpha,
        object_alpha=args.object_alpha,
        draw_scale_slices=args.draw_scale_slices,
        draw_node_traces=args.draw_node_traces,
        trace_alpha=args.trace_alpha,
        trace_limit=args.trace_limit,
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        dpi=args.dpi,
        id_col=args.id_col,
        only_object=args.only_object,
        no_background_cloud=args.no_background_cloud,
        object_color_by=args.object_color_by,
    )

    html_path = None
    if args.html:
        html_path = png_path.with_suffix(".html")
        html_trace_limit = args.html_trace_limit
        if html_trace_limit is None:
            html_trace_limit = args.trace_limit

        write_plotly_html(
            df=df,
            case=args.case,
            field=args.field,
            obj=args.object,
            outpath=html_path,
            title=args.title,
            id_col=args.id_col,
            only_object=args.only_object,
            no_background_cloud=args.no_background_cloud,
            object_color_by=args.object_color_by,
            html_node_traces=args.html_node_traces,
            html_scale_slices=args.html_scale_slices,
            html_trace_limit=html_trace_limit,
            html_marker_size=args.html_marker_size,
            html_object_marker_size=args.html_object_marker_size,
        )

    write_manifest(
        args=args,
        outdir=args.outdir,
        png_path=png_path,
        html_path=html_path,
        dyn_cols=dyn_cols,
        obj_cols=obj_cols,
        dyn_status=dyn_status,
        obj_status=obj_status,
        n_rows_total=len(df),
        n_rows_plotted=len(plotted_df),
    )

    print("OBS-076 scale-space stack figure complete")
    print("wrote:", png_path)
    if html_path is not None:
        print("wrote:", html_path)
    print("wrote:", args.outdir / "figure_obs076_scale_space_stack_manifest.csv")


if __name__ == "__main__":
    main()
