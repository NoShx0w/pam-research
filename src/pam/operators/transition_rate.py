"""Canonical transition-rate stage for the PAM operators layer."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_transition_targets(df: pd.DataFrame, within_k: int) -> pd.DataFrame:
    req = ["probe_id", "step", "lazarus_score", "distance_to_seam", "signed_phase"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    for probe_id, g in df.groupby("probe_id"):
        g = g.sort_values("step").reset_index(drop=True)

        prev_sign = 0
        flip_steps = []
        for _, row in g.iterrows():
            val = float(row["signed_phase"])
            sign = -1 if val < 0 else (1 if val > 0 else 0)
            if sign == 0:
                continue
            if prev_sign != 0 and sign != prev_sign:
                flip_steps.append(int(row["step"]))
            prev_sign = sign

        flip_steps = sorted(set(flip_steps))

        for _, row in g.iterrows():
            step = int(row["step"])
            future_flips = [fs for fs in flip_steps if fs > step]
            next_flip_step = future_flips[0] if future_flips else pd.NA

            transition_within_k = 0
            lag_to_next_transition = pd.NA
            if future_flips:
                lag = next_flip_step - step
                lag_to_next_transition = int(lag)
                transition_within_k = int(lag <= within_k)

            rows.append(
                {
                    "probe_id": probe_id,
                    "family": row["family"] if "family" in row else "",
                    "step": step,
                    "lazarus_score": float(row["lazarus_score"]) if pd.notna(row["lazarus_score"]) else float("nan"),
                    "distance_to_seam": float(row["distance_to_seam"]) if pd.notna(row["distance_to_seam"]) else float("nan"),
                    "scalar_curvature": float(row["scalar_curvature"]) if pd.notna(row["scalar_curvature"]) else float("nan"),
                    "signed_phase": float(row["signed_phase"]) if pd.notna(row["signed_phase"]) else float("nan"),
                    "next_flip_step": next_flip_step,
                    "lag_to_next_transition": lag_to_next_transition,
                    "transition_within_k": transition_within_k,
                }
            )

    return pd.DataFrame(rows)


def summarize_transition_rate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    med = work["lazarus_score"].median()
    work["lazarus_group"] = work["lazarus_score"].apply(
        lambda x: "high" if pd.notna(x) and x >= med else "low"
    )

    summary = (
        work.groupby("lazarus_group", as_index=False)
        .agg(
            n_states=("probe_id", "count"),
            transition_rate=("transition_within_k", "mean"),
            mean_lag_to_next_transition=("lag_to_next_transition", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            mean_distance_to_seam=("distance_to_seam", "mean"),
            mean_curvature=("scalar_curvature", "mean"),
        )
    )
    return work, summary


def render_plots(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path, within_k: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(summary["lazarus_group"], summary["transition_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"P(transition within {within_k} steps)")
    ax.set_title("Transition rate by Lazarus exposure")
    fig.tight_layout()
    fig.savefig(outdir / "transition_rate_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    plot_df = df[["lazarus_score", "lag_to_next_transition"]].copy()
    plot_df["lazarus_score"] = pd.to_numeric(plot_df["lazarus_score"], errors="coerce")
    plot_df["lag_to_next_transition"] = pd.to_numeric(plot_df["lag_to_next_transition"], errors="coerce")
    plot_df = plot_df.dropna()

    ax.scatter(plot_df["lazarus_score"], plot_df["lag_to_next_transition"], alpha=0.3)
    ax.set_xlabel("lazarus_score")
    ax.set_ylabel("lag_to_next_transition")
    ax.set_title("Transition lag vs Lazarus score")
    fig.tight_layout()
    fig.savefig(outdir / "transition_lag_scatter.png", dpi=220)
    plt.close(fig)


def run_transition_rate(
    paths_csv,
    outdir,
    within_k: int = 2,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(paths_csv)
    state_df = build_transition_targets(paths, within_k=within_k)
    labeled_df, summary = summarize_transition_rate(state_df)

    state_df.to_csv(outdir / "transition_rate_states.csv", index=False)
    labeled_df.to_csv(outdir / "transition_rate_labeled.csv", index=False)
    summary.to_csv(outdir / "transition_rate_summary.csv", index=False)

    render_plots(labeled_df, summary, outdir, within_k=within_k)

    print(outdir / "transition_rate_states.csv")
    print(outdir / "transition_rate_labeled.csv")
    print(outdir / "transition_rate_summary.csv")
    print(outdir / "transition_rate_bar.png")
    print(outdir / "transition_lag_scatter.png")

    return {
        "states": state_df,
        "labeled": labeled_df,
        "summary": summary,
    }
