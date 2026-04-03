"""
FIM Identity Trisurf Panel v2 — Canonical Data Version

Renders aligned trisurf plots for:
- identity magnitude (metric layer)
- signed local obstruction (connection layer)
- absolute holonomy / unsigned obstruction (invariant-derived layer)
- legacy spin (comparison)

Improvements over v1:
- merges canonical node + obstruction + MDS files
- masks long triangles to avoid ugly bridging across sparse regions
- synchronizes camera angle and XY limits across panels
- handles missing canonical columns gracefully

Canonical inputs:
- outputs/fim_identity/identity_field_nodes.csv
- outputs/fim_identity_obstruction/identity_obstruction_signed_nodes.csv
- outputs/fim_identity_obstruction/identity_obstruction_nodes.csv   (optional)
- outputs/fim_mds/mds_coords.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation


# ----------------------------
# Config
# ----------------------------

IDENTITY_NODES_CSV = Path("outputs/fim_identity/identity_field_nodes.csv")
OBSTRUCTION_SIGNED_CSV = Path("outputs/fim_identity_obstruction/identity_obstruction_signed_nodes.csv")
OBSTRUCTION_CSV = Path("outputs/fim_identity_obstruction/identity_obstruction_nodes.csv")
MDS_CSV = Path("outputs/fim_mds/mds_coords.csv")

OUTPUT_PATH = Path("outputs/fim_identity/identity_trisurf_panel_v2.png")

FIGSIZE = (16, 10)
VIEW_ELEV = 28
VIEW_AZIM = -58

# Relative mask threshold:
# triangles with an edge longer than this multiple of the median edge length are removed.
TRIANGLE_EDGE_MULTIPLIER = 2.75


# ----------------------------
# Load + merge canonical data
# ----------------------------

def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


require(IDENTITY_NODES_CSV)
require(OBSTRUCTION_SIGNED_CSV)
require(MDS_CSV)

identity_df = pd.read_csv(IDENTITY_NODES_CSV)
obstruction_signed_df = pd.read_csv(OBSTRUCTION_SIGNED_CSV)
obstruction_df = pd.read_csv(OBSTRUCTION_CSV) if OBSTRUCTION_CSV.exists() else pd.DataFrame()
mds_df = pd.read_csv(MDS_CSV)

for df in [identity_df, obstruction_signed_df, obstruction_df, mds_df]:
    if "node_id" in df.columns:
        df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("Int64").astype(str)

identity_keep = [c for c in ["node_id", "identity_magnitude", "identity_spin"] if c in identity_df.columns]
signed_keep = [
    c for c in [
        "node_id",
        "obstruction_mean_holonomy",
        "obstruction_mean_abs_holonomy",
        "obstruction_signed_sum_holonomy",
        "obstruction_signed_weighted_holonomy",
        "obstruction_max_abs_holonomy",
    ]
    if c in obstruction_signed_df.columns
]
unsigned_keep = [
    c for c in [
        "node_id",
        "obstruction_mean_abs_holonomy",
        "obstruction_max_abs_holonomy",
        "absolute_holonomy_node",
    ]
    if c in obstruction_df.columns
]
mds_keep = [c for c in ["node_id", "mds1", "mds2"] if c in mds_df.columns]

df = identity_df[identity_keep].copy()

if signed_keep:
    df = df.merge(obstruction_signed_df[signed_keep], on="node_id", how="left")

if unsigned_keep:
    df = df.merge(
        obstruction_df[unsigned_keep],
        on="node_id",
        how="left",
        suffixes=("", "_unsigned"),
    )

df = df.merge(mds_df[mds_keep], on="node_id", how="left")

required = {"mds1", "mds2", "identity_magnitude"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Merged dataframe is missing required columns: {sorted(missing)}")

# Canonical signed local obstruction
if "obstruction_signed_sum_holonomy" in df.columns:
    df["identity_obstruction"] = pd.to_numeric(df["obstruction_signed_sum_holonomy"], errors="coerce")
elif "obstruction_signed_weighted_holonomy" in df.columns:
    df["identity_obstruction"] = pd.to_numeric(df["obstruction_signed_weighted_holonomy"], errors="coerce")
elif "obstruction_mean_holonomy" in df.columns:
    df["identity_obstruction"] = pd.to_numeric(df["obstruction_mean_holonomy"], errors="coerce")
else:
    df["identity_obstruction"] = pd.to_numeric(df.get("identity_spin"), errors="coerce")

# Canonical absolute holonomy / unsigned obstruction
if "absolute_holonomy_node" in df.columns:
    df["holonomy"] = pd.to_numeric(df["absolute_holonomy_node"], errors="coerce")
elif "obstruction_mean_abs_holonomy" in df.columns:
    df["holonomy"] = pd.to_numeric(df["obstruction_mean_abs_holonomy"], errors="coerce")
else:
    df["holonomy"] = np.abs(pd.to_numeric(df.get("identity_spin"), errors="coerce"))

df["identity_spin"] = pd.to_numeric(df.get("identity_spin"), errors="coerce")
df["identity_magnitude"] = pd.to_numeric(df["identity_magnitude"], errors="coerce")
df["mds1"] = pd.to_numeric(df["mds1"], errors="coerce")
df["mds2"] = pd.to_numeric(df["mds2"], errors="coerce")

plot_df = df.dropna(
    subset=[
        "mds1",
        "mds2",
        "identity_magnitude",
        "identity_obstruction",
        "holonomy",
        "identity_spin",
    ]
).copy()

if len(plot_df) < 3:
    raise ValueError("Not enough valid rows to build a trisurf panel.")


# ----------------------------
# Shared triangulation + masking
# ----------------------------

x = plot_df["mds1"].to_numpy(dtype=float)
y = plot_df["mds2"].to_numpy(dtype=float)

z_mag = plot_df["identity_magnitude"].to_numpy(dtype=float)
z_obs = plot_df["identity_obstruction"].to_numpy(dtype=float)
z_hol = plot_df["holonomy"].to_numpy(dtype=float)
z_spin = plot_df["identity_spin"].to_numpy(dtype=float)

tri = Triangulation(x, y)


def triangle_edge_lengths(triangles: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = np.stack([x, y], axis=1)
    lengths = np.zeros((len(triangles), 3), dtype=float)

    for k, (i, j, m) in enumerate(triangles):
        p0, p1, p2 = pts[i], pts[j], pts[m]
        lengths[k, 0] = np.linalg.norm(p0 - p1)
        lengths[k, 1] = np.linalg.norm(p1 - p2)
        lengths[k, 2] = np.linalg.norm(p2 - p0)

    return lengths


edge_lengths = triangle_edge_lengths(tri.triangles, x, y)
median_edge = float(np.nanmedian(edge_lengths))
if not np.isfinite(median_edge) or median_edge <= 0:
    median_edge = 1.0

mask_threshold = TRIANGLE_EDGE_MULTIPLIER * median_edge
triangle_mask = np.max(edge_lengths, axis=1) > mask_threshold
tri.set_mask(triangle_mask)


# ----------------------------
# Plot helper
# ----------------------------

def plot_surface(ax, z: np.ndarray, title: str, cmap: str = "viridis", diverging: bool = False):
    if diverging:
        vmax = float(np.nanmax(np.abs(z)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin, vmax = None, None

    surf = ax.plot_trisurf(
        tri,
        z,
        cmap=cmap,
        linewidth=0.15,
        antialiased=True,
        shade=True,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.set_zlabel("Z")
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

    return surf


# ----------------------------
# Figure layout
# ----------------------------

fig = plt.figure(figsize=FIGSIZE)

axes = [
    fig.add_subplot(2, 2, 1, projection="3d"),
    fig.add_subplot(2, 2, 2, projection="3d"),
    fig.add_subplot(2, 2, 3, projection="3d"),
    fig.add_subplot(2, 2, 4, projection="3d"),
]

plot_surface(
    axes[0],
    z_mag,
    "Identity Magnitude (Metric Layer)",
    cmap="viridis",
)

plot_surface(
    axes[1],
    z_obs,
    "Signed Local Obstruction (Connection Layer)",
    cmap="coolwarm",
    diverging=True,
)

plot_surface(
    axes[2],
    z_hol,
    "Absolute Holonomy / Unsigned Obstruction",
    cmap="plasma",
)

plot_surface(
    axes[3],
    z_spin,
    "Legacy Spin (Comparison)",
    cmap="coolwarm",
    diverging=True,
)


# ----------------------------
# Finalize
# ----------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=220)
plt.close()

print(f"Saved: {OUTPUT_PATH}")