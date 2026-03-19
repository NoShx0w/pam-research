#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd


def load_node_table(
    mds_csv: str | Path,
    phase_csv: str | Path,
    lazarus_csv: str | Path,
) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    phase = pd.read_csv(phase_csv)
    laz = pd.read_csv(lazarus_csv)

    df = mds.copy()
    keep_phase = [c for c in ["r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]
    df = df.merge(phase[keep_phase], on=["r", "alpha"], how="left")
    keep_laz = [c for c in ["r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns]
    df = df.merge(laz[keep_laz], on=["r", "alpha"], how="left")
    return df


def load_seam(seam_csv: str | Path, node_df: pd.DataFrame) -> pd.DataFrame:
    p = Path(seam_csv)
    if not p.exists():
        return pd.DataFrame()
    seam = pd.read_csv(p)
    if not {"mds1", "mds2"}.issubset(seam.columns):
        seam = seam.merge(node_df[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")
    return seam.dropna(subset=["mds1", "mds2"]).copy()


def load_paths(paths_csv: str | Path) -> pd.DataFrame:
    p = Path(paths_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing paths CSV: {p}")
    return pd.read_csv(p)


def choose_path(paths: pd.DataFrame, probe_id: str | None = None) -> tuple[str, pd.DataFrame]:
    if probe_id is not None:
        grp = paths[paths["probe_id"] == probe_id].copy()
        if grp.empty:
            raise ValueError(f"probe_id '{probe_id}' not found in paths CSV")
        return probe_id, grp.sort_values("step").reset_index(drop=True)

    summaries = []
    for pid, grp in paths.groupby("probe_id"):
        grp = grp.sort_values("step")
        laz_max = float(grp["lazarus_score"].max()) if "lazarus_score" in grp.columns else 0.0
        flip = 0
        prev = 0
        for v in grp["signed_phase"].astype(float):
            s = -1 if v < 0 else (1 if v > 0 else 0)
            if s == 0:
                continue
            if prev != 0 and s != prev:
                flip = 1
                break
            prev = s
        summaries.append({"probe_id": pid, "flip": flip, "laz_max": laz_max, "n": len(grp)})
    sdf = pd.DataFrame(summaries).sort_values(["flip", "laz_max", "n"], ascending=[False, False, False])
    best = str(sdf.iloc[0]["probe_id"])
    grp = paths[paths["probe_id"] == best].sort_values("step").reset_index(drop=True)
    return best, grp


def find_transition_index(grp: pd.DataFrame) -> int | None:
    prev = 0
    for i, v in enumerate(grp["signed_phase"].astype(float)):
        s = -1 if v < 0 else (1 if v > 0 else 0)
        if s == 0:
            continue
        if prev != 0 and s != prev:
            return i
        prev = s
    return None


def find_lazarus_peak_index(grp: pd.DataFrame) -> int | None:
    if "lazarus_score" not in grp.columns:
        return None
    return int(grp["lazarus_score"].astype(float).idxmax())


def build_animation(
    nodes: pd.DataFrame,
    seam: pd.DataFrame,
    grp: pd.DataFrame,
    outpath: Path,
    fps: int = 6,
    pause_frames: int = 6,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=nodes["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=36,
        alpha=0.45,
    )
    plt.colorbar(sc, ax=ax, shrink=0.84, label="signed phase")

    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.4, alpha=0.95, label="seam")

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("PAM Observatory — Phase trajectory")
    ax.legend(loc="best")

    ax.set_xlim(nodes["mds1"].min() - 0.3, nodes["mds1"].max() + 0.3)
    ax.set_ylim(nodes["mds2"].min() - 0.3, nodes["mds2"].max() + 0.3)

    path_line, = ax.plot([], [], linewidth=2.6, alpha=0.95)
    current_pt = ax.scatter([], [], s=140, edgecolors="black", linewidths=0.8, zorder=5)
    start_pt = ax.scatter([], [], s=90, marker="o", zorder=5)
    peak_pt = ax.scatter([], [], s=140, marker="o", edgecolors="black", linewidths=0.8, zorder=6)
    flip_pt = ax.scatter([], [], s=140, marker="X", zorder=6)
    annotation = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )

    peak_idx = find_lazarus_peak_index(grp)
    flip_idx = find_transition_index(grp)

    base_frames = list(range(len(grp)))
    extra = []
    if flip_idx is not None:
        extra.extend([flip_idx] * pause_frames)
    if len(grp) > 0:
        extra.extend([len(grp) - 1] * pause_frames)
    frames = base_frames + extra

    def init():
        path_line.set_data([], [])
        current_pt.set_offsets([[float("nan"), float("nan")]])
        start_pt.set_offsets([[float("nan"), float("nan")]])
        peak_pt.set_offsets([[float("nan"), float("nan")]])
        flip_pt.set_offsets([[float("nan"), float("nan")]])
        annotation.set_text("")
        return path_line, current_pt, start_pt, peak_pt, flip_pt, annotation

    def update(frame_id: int):
        row = grp.iloc[frame_id]
        sub = grp.iloc[: frame_id + 1]

        path_line.set_data(sub["mds1"], sub["mds2"])

        cur_xy = [[row["mds1"], row["mds2"]]]
        current_pt.set_offsets(cur_xy)
        current_pt.set_array(pd.Series([row["lazarus_score"]]).astype(float))
        current_pt.set_cmap("plasma")

        start_xy = [[grp.iloc[0]["mds1"], grp.iloc[0]["mds2"]]]
        start_pt.set_offsets(start_xy)

        if peak_idx is not None and frame_id >= peak_idx:
            peak_row = grp.iloc[peak_idx]
            peak_pt.set_offsets([[peak_row["mds1"], peak_row["mds2"]]])
            peak_pt.set_facecolor("tab:green")
        else:
            peak_pt.set_offsets([[float("nan"), float("nan")]])

        if flip_idx is not None and frame_id >= flip_idx:
            flip_row = grp.iloc[flip_idx]
            flip_pt.set_offsets([[flip_row["mds1"], flip_row["mds2"]]])
            flip_pt.set_facecolor("tab:red")
        else:
            flip_pt.set_offsets([[float("nan"), float("nan")]])

        text = [
            f"probe_id: {grp.iloc[0]['probe_id']}",
            f"step: {int(row['step'])}",
            f"signed_phase: {float(row['signed_phase']):.3f}",
            f"distance_to_seam: {float(row['distance_to_seam']):.3f}",
            f"lazarus_score: {float(row['lazarus_score']):.3f}",
        ]

        if peak_idx is not None and frame_id == peak_idx:
            text.append("Lazarus peak")
        if flip_idx is not None and frame_id == flip_idx:
            text.append("phase transition")

        annotation.set_text("\n".join(text))
        return path_line, current_pt, start_pt, peak_pt, flip_pt, annotation

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)

    if outpath.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2200)

    ani.save(outpath, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate a phase trajectory on the PAM manifold.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--probe-id", default=None, help="Optional explicit probe_id to animate")
    parser.add_argument("--out", default="outputs/fim_anim/phase_trajectory.gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--pause-frames", type=int, default=6)
    args = parser.parse_args()

    nodes = load_node_table(args.mds_csv, args.phase_csv, args.lazarus_csv)
    seam = load_seam(args.seam_csv, nodes)
    paths = load_paths(args.paths_csv)
    probe_id, grp = choose_path(paths, args.probe_id)

    outpath = Path(args.out)
    build_animation(nodes, seam, grp, outpath, fps=args.fps, pause_frames=args.pause_frames)

    print(f"Animated probe_id: {probe_id}")
    print(outpath)


if __name__ == "__main__":
    main()
