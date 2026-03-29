import argparse
from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def invert_distance(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return 1.0 / (1.0 + s.clip(lower=0.0))


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    finite = s.dropna()
    if len(finite) == 0:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    return ((s - vmin) / (vmax - vmin)).fillna(0.0).clip(0.0, 1.0)


def summarize_predictor(df: pd.DataFrame, predictor_col: str, label: str) -> dict:
    work = df[[predictor_col, "transition_within_k"]].copy()
    work[predictor_col] = pd.to_numeric(work[predictor_col], errors="coerce")
    work["transition_within_k"] = pd.to_numeric(work["transition_within_k"], errors="coerce")
    work = work.dropna()

    if len(work) == 0:
        return {
            "predictor": label,
            "n_states": 0,
            "transition_rate_high": pd.NA,
            "transition_rate_low": pd.NA,
            "rate_ratio": pd.NA,
            "rate_diff": pd.NA,
            "median_threshold": pd.NA,
        }

    thr = float(work[predictor_col].median())
    high = work[work[predictor_col] >= thr]
    low = work[work[predictor_col] < thr]

    high_rate = float(high["transition_within_k"].mean()) if len(high) else pd.NA
    low_rate = float(low["transition_within_k"].mean()) if len(low) else pd.NA

    ratio = pd.NA
    diff = pd.NA
    if pd.notna(high_rate) and pd.notna(low_rate):
        diff = high_rate - low_rate
        ratio = high_rate / low_rate if low_rate != 0 else pd.NA

    return {
        "predictor": label,
        "n_states": int(len(work)),
        "transition_rate_high": high_rate,
        "transition_rate_low": low_rate,
        "rate_ratio": ratio,
        "rate_diff": diff,
        "median_threshold": thr,
    }


def summarize_joint_model(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["distance_to_seam", "scalar_curvature", "lazarus_score", "transition_within_k"]].copy()
    work = safe_numeric(work, ["distance_to_seam", "scalar_curvature", "lazarus_score", "transition_within_k"])
    work = work.dropna()

    if len(work) == 0:
        return pd.DataFrame()

    work["inv_distance"] = invert_distance(work["distance_to_seam"])
    work["log_curvature"] = work["scalar_curvature"].clip(lower=0.0).map(lambda x: math.log10(1.0 + x))
    work["inv_distance_norm"] = normalize_series(work["inv_distance"])
    work["log_curvature_norm"] = normalize_series(work["log_curvature"])
    work["lazarus_score_norm"] = normalize_series(work["lazarus_score"])

    work["baseline_score"] = 0.5 * work["inv_distance_norm"] + 0.5 * work["log_curvature_norm"]
    work["baseline_plus_lazarus"] = (
        0.4 * work["inv_distance_norm"]
        + 0.3 * work["log_curvature_norm"]
        + 0.3 * work["lazarus_score_norm"]
    )

    rows = [
        summarize_predictor(work, "baseline_score", "distance+curvature"),
        summarize_predictor(work, "baseline_plus_lazarus", "distance+curvature+lazarus"),
    ]
    return pd.DataFrame(rows)


def render_plot(summary: pd.DataFrame, outpath: Path) -> None:
    plot_df = summary.copy()
    plot_df["rate_ratio"] = pd.to_numeric(plot_df["rate_ratio"], errors="coerce")
    plot_df["rate_diff"] = pd.to_numeric(plot_df["rate_diff"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    rr = plot_df.dropna(subset=["rate_ratio"])
    axes[0].bar(rr["predictor"], rr["rate_ratio"])
    axes[0].set_ylabel("rate ratio (high / low)")
    axes[0].set_title("Predictive separation by observable")
    axes[0].tick_params(axis="x", rotation=20)

    rd = plot_df.dropna(subset=["rate_diff"])
    axes[1].bar(rd["predictor"], rd["rate_diff"])
    axes[1].set_ylabel("rate difference (high - low)")
    axes[1].set_title("Transition-rate difference by observable")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ablation of Lazarus against simple predictor baselines.")
    parser.add_argument("--transition-labeled-csv", default="outputs/fim_transition_rate/transition_rate_labeled.csv")
    parser.add_argument("--outdir", default="outputs/fim_ablation")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    df = safe_numeric(df, ["lazarus_score", "distance_to_seam", "scalar_curvature", "transition_within_k"])

    df["inverse_distance"] = invert_distance(df["distance_to_seam"])
    df["log_curvature"] = df["scalar_curvature"].clip(lower=0.0).map(lambda x: math.log10(1.0 + x))
    if "constraint_strength" not in df.columns:
        d = df["distance_to_seam"].clip(lower=0.0).fillna(0.0)
        c = df["scalar_curvature"].clip(lower=0.0).fillna(0.0)
        df["constraint_strength"] = c / (d + 1e-6)

    rows = [
        summarize_predictor(df, "lazarus_score", "lazarus"),
        summarize_predictor(df, "inverse_distance", "inverse_distance"),
        summarize_predictor(df, "log_curvature", "log_curvature"),
        summarize_predictor(df, "constraint_strength", "constraint_strength"),
    ]

    summary = pd.DataFrame(rows)
    joint = summarize_joint_model(df)
    full = pd.concat([summary, joint], ignore_index=True)

    full.to_csv(outdir / "lazarus_ablation_summary.csv", index=False)
    render_plot(full, outdir / "lazarus_ablation_effects.png")

    print(outdir / "lazarus_ablation_summary.csv")
    print(outdir / "lazarus_ablation_effects.png")


if __name__ == "__main__":
    main()
