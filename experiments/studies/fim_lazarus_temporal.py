import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_temporal_metrics(df: pd.DataFrame) -> pd.DataFrame:
    req = ["probe_id", "step", "lazarus_score", "distance_to_seam", "signed_phase"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    for probe_id, g in df.groupby("probe_id"):
        g = g.sort_values("step").reset_index(drop=True)

        laz_idx = int(g["lazarus_score"].astype(float).idxmax())
        laz_step = int(g.loc[laz_idx, "step"])
        laz_value = float(g.loc[laz_idx, "lazarus_score"])

        d2s = g["distance_to_seam"].astype(float)
        seam_idx = int(d2s.idxmin())
        seam_step = int(g.loc[seam_idx, "step"])
        seam_distance = float(g.loc[seam_idx, "distance_to_seam"])

        signed = g["signed_phase"].astype(float).tolist()
        phase_flip_step = None
        prev_sign = 0
        for _, row in g.iterrows():
            val = float(row["signed_phase"])
            sign = -1 if val < 0 else (1 if val > 0 else 0)
            if sign == 0:
                continue
            if prev_sign != 0 and sign != prev_sign:
                phase_flip_step = int(row["step"])
                break
            prev_sign = sign

        if phase_flip_step is None:
            lag_to_flip = pd.NA
            lazarus_precedes_flip = 0
        else:
            lag_to_flip = int(phase_flip_step - laz_step)
            lazarus_precedes_flip = int(laz_step <= phase_flip_step)

        lag_to_seam = int(seam_step - laz_step)
        lazarus_precedes_seam = int(laz_step <= seam_step)

        rows.append(
            {
                "probe_id": probe_id,
                "family": g["family"].iloc[0] if "family" in g.columns else "",
                "num_steps": int(len(g)),
                "lazarus_peak_step": laz_step,
                "lazarus_peak_value": laz_value,
                "seam_contact_step": seam_step,
                "seam_contact_distance": seam_distance,
                "phase_flip_step": phase_flip_step if phase_flip_step is not None else pd.NA,
                "lag_lazarus_to_seam": lag_to_seam,
                "lag_lazarus_to_flip": lag_to_flip,
                "lazarus_precedes_seam": lazarus_precedes_seam,
                "lazarus_precedes_flip": lazarus_precedes_flip,
            }
        )

    return pd.DataFrame(rows)


def build_summary(tdf: pd.DataFrame) -> pd.DataFrame:
    def mean_or_nan(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        return s.mean()

    summary = pd.DataFrame(
        {
            "n_paths": [int(len(tdf))],
            "share_lazarus_precedes_seam": [tdf["lazarus_precedes_seam"].mean()],
            "share_lazarus_precedes_flip": [tdf["lazarus_precedes_flip"].mean()],
            "mean_lag_lazarus_to_seam": [mean_or_nan(tdf["lag_lazarus_to_seam"])],
            "mean_lag_lazarus_to_flip": [mean_or_nan(tdf["lag_lazarus_to_flip"])],
            "median_lag_lazarus_to_seam": [pd.to_numeric(tdf["lag_lazarus_to_seam"], errors="coerce").median()],
            "median_lag_lazarus_to_flip": [pd.to_numeric(tdf["lag_lazarus_to_flip"], errors="coerce").median()],
        }
    )
    return summary


def render_plots(tdf: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # scatter: lazarus peak vs seam contact
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(tdf["lazarus_peak_step"], tdf["seam_contact_step"], alpha=0.8)
    mn = min(tdf["lazarus_peak_step"].min(), tdf["seam_contact_step"].min())
    mx = max(tdf["lazarus_peak_step"].max(), tdf["seam_contact_step"].max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2)
    ax.set_xlabel("lazarus_peak_step")
    ax.set_ylabel("seam_contact_step")
    ax.set_title("Temporal ordering: Lazarus peak vs seam contact")
    fig.tight_layout()
    fig.savefig(outdir / "lazarus_temporal_scatter.png", dpi=220)
    plt.close(fig)

    # histogram of lags
    lag = pd.to_numeric(tdf["lag_lazarus_to_seam"], errors="coerce").dropna()
    if len(lag):
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.hist(lag, bins=min(20, max(5, len(lag) // 3)))
        ax.set_xlabel("lag_lazarus_to_seam")
        ax.set_ylabel("count")
        ax.set_title("Lag from Lazarus peak to seam contact")
        fig.tight_layout()
        fig.savefig(outdir / "lazarus_temporal_lag_hist.png", dpi=220)
        plt.close(fig)

    # family summary
    if "family" in tdf.columns:
        fam = (
            tdf.groupby("family", as_index=False)
            .agg(
                n_paths=("probe_id", "count"),
                share_lazarus_precedes_seam=("lazarus_precedes_seam", "mean"),
                mean_lag_lazarus_to_seam=("lag_lazarus_to_seam", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            )
        )
        fam.to_csv(outdir / "lazarus_temporal_by_family.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Temporal analysis of Lazarus peak ordering relative to seam contact and phase flip."
    )
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--outdir", default="outputs/fim_lazarus_temporal")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.paths_csv)
    tdf = build_temporal_metrics(df)
    summary = build_summary(tdf)

    tdf.to_csv(outdir / "lazarus_temporal_metrics.csv", index=False)
    summary.to_csv(outdir / "lazarus_temporal_summary.csv", index=False)

    render_plots(tdf, outdir)

    print(outdir / "lazarus_temporal_metrics.csv")
    print(outdir / "lazarus_temporal_summary.csv")
    print(outdir / "lazarus_temporal_scatter.png")
    print(outdir / "lazarus_temporal_lag_hist.png")


if __name__ == "__main__":
    main()
