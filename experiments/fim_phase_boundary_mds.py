import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def greedy_order(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n <= 2:
        return np.arange(n)

    # start from the leftmost point in MDS1
    start = int(np.argmin(points[:, 0]))
    unused = set(range(n))
    order = [start]
    unused.remove(start)

    while unused:
        last = order[-1]
        nxt = min(unused, key=lambda j: np.sum((points[j] - points[last]) ** 2))
        order.append(nxt)
        unused.remove(nxt)

    return np.array(order, dtype=int)


def densify_polyline(points: np.ndarray, n_samples: int = 100) -> np.ndarray:
    if len(points) == 1:
        return points.copy()

    seglens = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    total = float(np.sum(seglens))
    if total == 0:
        return np.repeat(points[:1], n_samples, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seglens)])
    ts = np.linspace(0.0, total, n_samples)

    out = []
    for t in ts:
        k = np.searchsorted(cum, t, side="right") - 1
        k = min(max(k, 0), len(seglens) - 1)
        t0, t1 = cum[k], cum[k + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p = (1 - w) * points[k] + w * points[k + 1]
        out.append(p)

    return np.asarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-csv", default="outputs/fim_phase/phase_boundary_points.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--outdir", default="outputs/fim_phase")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    boundary = pd.read_csv(args.boundary_csv)
    mds = pd.read_csv(args.mds_csv)

    seam = boundary.merge(mds, on=["r", "alpha"], how="left")
    seam = seam.dropna(subset=["mds1", "mds2"]).copy()

    if seam.empty:
        raise ValueError("No seam points could be matched to MDS coordinates.")

    pts = seam[["mds1", "mds2"]].to_numpy(dtype=float)
    order = greedy_order(pts)
    seam_ord = seam.iloc[order].reset_index(drop=True)
    pts_ord = seam_ord[["mds1", "mds2"]].to_numpy(dtype=float)

    curve = densify_polyline(pts_ord, n_samples=args.n_samples)

    # back-project each fitted point to nearest manifold node in MDS space
    all_pts = mds[["mds1", "mds2"]].to_numpy(dtype=float)
    nearest_idx = []
    for p in curve:
        j = int(np.argmin(np.sum((all_pts - p) ** 2, axis=1)))
        nearest_idx.append(j)

    back = mds.iloc[nearest_idx].copy()
    back = back.drop_duplicates(subset=["r", "alpha"]).reset_index(drop=True)

    back_csv = outdir / "phase_boundary_mds_backprojected.csv"
    back.to_csv(back_csv, index=False)

    # plot seam in MDS space
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(mds["mds1"], mds["mds2"], s=24, alpha=0.35)
    ax.scatter(seam_ord["mds1"], seam_ord["mds2"], s=80)
    ax.plot(curve[:, 0], curve[:, 1], linewidth=2.5)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Phase seam fitted in MDS space")
    fig.tight_layout()
    fig.savefig(outdir / "phase_boundary_mds.png", dpi=180)
    plt.close(fig)

    # plot backprojected boundary in parameter space
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(mds["alpha"], mds["r"], s=20, alpha=0.25)
    ax.scatter(back["alpha"], back["r"], s=80)
    ax.plot(back["alpha"], back["r"], linewidth=2.5)
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title("Phase boundary backprojected from MDS seam")
    fig.tight_layout()
    fig.savefig(outdir / "phase_boundary_mds_backprojected.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {back_csv}")
    print(f"Wrote {outdir / 'phase_boundary_mds.png'}")
    print(f"Wrote {outdir / 'phase_boundary_mds_backprojected.png'}")


if __name__ == "__main__":
    main()
