import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clamp01(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=1.0)


def safe_normalize(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    finite = s[pd.notna(s)]
    if len(finite) == 0:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    return ((s - vmin) / (vmax - vmin)).fillna(0.0).clip(0.0, 1.0)


def build_horizon_metrics(df: pd.DataFrame, lazarus_threshold: float | None = None) -> pd.DataFrame:
    out = df.copy()

    out["distance_to_horizon"] = out["min_distance_to_seam"].astype(float)
    out["horizon_pressure"] = out["constraint_strength"].astype(float)
    out["collapse_event"] = out["crosses_seam"].fillna(0).astype(int)
    out["phase_flip_event"] = (out["phase_flip_count"].fillna(0) > 0).astype(int)
    out["precollapse_signal"] = out["lazarus_max"].astype(float)
    out["precollapse_hit"] = out["lazarus_hit_any"].fillna(0).astype(int)

    if lazarus_threshold is None:
        lazarus_threshold = float(out["precollapse_signal"].median())

    out["precollapse_threshold"] = lazarus_threshold
    out["precollapse_signal_hit"] = (out["precollapse_signal"] >= lazarus_threshold).astype(int)

    out["distance_to_horizon_norm"] = 1.0 - safe_normalize(out["distance_to_horizon"])
    out["horizon_pressure_norm"] = safe_normalize(out["horizon_pressure"])
    out["curvature_norm"] = safe_normalize(out["max_curvature_along_path"].fillna(0.0))
    out["path_length_norm"] = safe_normalize(out["path_length_fisher"].fillna(0.0))
    out["phase_flip_norm"] = safe_normalize(out["phase_flip_count"].fillna(0.0))

    out["collapse_risk"] = clamp01(
        0.30 * out["distance_to_horizon_norm"]
        + 0.25 * out["horizon_pressure_norm"]
        + 0.20 * out["curvature_norm"]
        + 0.15 * out["phase_flip_norm"]
        + 0.10 * safe_normalize(out["precollapse_signal"].fillna(0.0))
    )

    labels = pd.Series(["open"] * len(out), index=out.index, dtype="string")
    labels[(out["precollapse_hit"] == 1) & (out["collapse_event"] == 0)] = "precollapse"
    labels[(out["collapse_event"] == 1) & (out["phase_flip_event"] == 0)] = "boundary_contact"
    labels[(out["collapse_event"] == 1) & (out["phase_flip_event"] == 1)] = "collapse_transition"
    out["horizon_type"] = labels

    return out


def summarize_predictive(out: pd.DataFrame) -> pd.DataFrame:
    groups = out["precollapse_hit"].map({1: "precollapse_hit", 0: "no_precollapse_hit"})
    return (
        out.groupby(groups, as_index=False)
        .agg(
            n_paths=("probe_id", "count"),
            seam_cross_rate=("collapse_event", "mean"),
            mean_phase_flip_count=("phase_flip_count", "mean"),
            mean_min_distance_to_horizon=("distance_to_horizon", "mean"),
            mean_max_curvature=("max_curvature_along_path", "mean"),
            mean_path_length=("path_length_fisher", "mean"),
            mean_horizon_pressure=("horizon_pressure", "mean"),
            mean_collapse_risk=("collapse_risk", "mean"),
        )
        .rename(columns={"precollapse_hit": "group"})
    )


def render_plots(out: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(out["precollapse_signal"], out["collapse_risk"], alpha=0.8)
    ax.set_xlabel("precollapse_signal (lazarus_max)")
    ax.set_ylabel("collapse_risk")
    ax.set_title("Collapse risk vs pre-collapse signal")
    fig.tight_layout()
    fig.savefig(outdir / "horizon_from_probes_scatter.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.bar(summary["group"], summary["seam_cross_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(collapse_event)")
    ax.set_title("Boundary interaction by pre-collapse regime")
    fig.tight_layout()
    fig.savefig(outdir / "horizon_from_probes_bar.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build operator-native horizon metrics from scaled probe outputs."
    )
    parser.add_argument("--input-csv", default="outputs/fim_ops_scaled/scaled_probe_metrics.csv")
    parser.add_argument("--outdir", default="outputs/fim_horizon")
    parser.add_argument(
        "--lazarus-threshold",
        type=float,
        default=None,
        help="Optional explicit threshold for precollapse_signal_hit; defaults to median lazarus_max.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    out = build_horizon_metrics(df, lazarus_threshold=args.lazarus_threshold)
    summary = summarize_predictive(out)

    out.to_csv(outdir / "horizon_metrics_from_probes.csv", index=False)
    summary.to_csv(outdir / "horizon_predictive_summary_from_probes.csv", index=False)

    render_plots(out, summary, outdir)

    print(outdir / "horizon_metrics_from_probes.csv")
    print(outdir / "horizon_predictive_summary_from_probes.csv")
    print(outdir / "horizon_from_probes_scatter.png")
    print(outdir / "horizon_from_probes_bar.png")


if __name__ == "__main__":
    main()
