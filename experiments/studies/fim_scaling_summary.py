import argparse
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def extract_transition_metrics(transition_summary: pd.DataFrame) -> dict:
    out = {
        "transition_rate_high": pd.NA,
        "transition_rate_low": pd.NA,
        "transition_rate_ratio": pd.NA,
        "transition_rate_diff": pd.NA,
        "mean_lag_high": pd.NA,
        "mean_lag_low": pd.NA,
        "mean_distance_high": pd.NA,
        "mean_distance_low": pd.NA,
        "mean_curvature_high": pd.NA,
        "mean_curvature_low": pd.NA,
    }

    if "lazarus_group" not in transition_summary.columns:
        return out

    high = transition_summary[transition_summary["lazarus_group"] == "high"]
    low = transition_summary[transition_summary["lazarus_group"] == "low"]

    if len(high):
        high = high.iloc[0]
        out["transition_rate_high"] = high.get("transition_rate", pd.NA)
        out["mean_lag_high"] = high.get("mean_lag_to_next_transition", pd.NA)
        out["mean_distance_high"] = high.get("mean_distance_to_seam", pd.NA)
        out["mean_curvature_high"] = high.get("mean_curvature", pd.NA)

    if len(low):
        low = low.iloc[0]
        out["transition_rate_low"] = low.get("transition_rate", pd.NA)
        out["mean_lag_low"] = low.get("mean_lag_to_next_transition", pd.NA)
        out["mean_distance_low"] = low.get("mean_distance_to_seam", pd.NA)
        out["mean_curvature_low"] = low.get("mean_curvature", pd.NA)

    if pd.notna(out["transition_rate_high"]) and pd.notna(out["transition_rate_low"]):
        low_val = float(out["transition_rate_low"])
        high_val = float(out["transition_rate_high"])
        out["transition_rate_diff"] = high_val - low_val
        out["transition_rate_ratio"] = high_val / low_val if low_val != 0 else pd.NA

    return out


def extract_horizon_metrics(horizon_summary: pd.DataFrame) -> dict:
    out = {
        "seam_cross_rate_precollapse": pd.NA,
        "seam_cross_rate_no_precollapse": pd.NA,
        "collapse_risk_precollapse": pd.NA,
        "collapse_risk_no_precollapse": pd.NA,
    }

    if "group" not in horizon_summary.columns:
        return out

    pre = horizon_summary[horizon_summary["group"] == "precollapse_hit"]
    no_pre = horizon_summary[horizon_summary["group"] == "no_precollapse_hit"]

    if len(pre):
        pre = pre.iloc[0]
        out["seam_cross_rate_precollapse"] = pre.get("seam_cross_rate", pd.NA)
        out["collapse_risk_precollapse"] = pre.get("mean_collapse_risk", pd.NA)

    if len(no_pre):
        no_pre = no_pre.iloc[0]
        out["seam_cross_rate_no_precollapse"] = no_pre.get("seam_cross_rate", pd.NA)
        out["collapse_risk_no_precollapse"] = no_pre.get("mean_collapse_risk", pd.NA)

    return out


def extract_temporal_metrics(temporal_summary: pd.DataFrame) -> dict:
    out = {
        "share_lazarus_precedes_seam": pd.NA,
        "share_lazarus_precedes_flip": pd.NA,
        "mean_lag_lazarus_to_seam": pd.NA,
        "mean_lag_lazarus_to_flip": pd.NA,
        "median_lag_lazarus_to_seam": pd.NA,
        "median_lag_lazarus_to_flip": pd.NA,
    }

    if len(temporal_summary) == 0:
        return out

    row = temporal_summary.iloc[0]
    for k in out:
        out[k] = row.get(k, pd.NA)

    return out


def summarize_scale(scale_dir: Path, probe_count: int) -> dict:
    transition_summary = read_csv(scale_dir / "fim_transition_rate" / "transition_rate_summary.csv")
    horizon_summary = read_csv(scale_dir / "fim_horizon" / "horizon_predictive_summary_from_probes.csv")
    temporal_summary = read_csv(scale_dir / "fim_lazarus_temporal" / "lazarus_temporal_summary.csv")

    row = {"probe_count": probe_count, "scale_dir": str(scale_dir)}
    row.update(extract_transition_metrics(transition_summary))
    row.update(extract_horizon_metrics(horizon_summary))
    row.update(extract_temporal_metrics(temporal_summary))
    return row


def parse_scale_arg(scale_arg: str) -> tuple[int, Path]:
    """
    Accept format:
      100=outputs/scales/100
    """
    if "=" not in scale_arg:
        raise ValueError(
            f"Invalid --scale value '{scale_arg}'. Expected format like: 100=outputs/scales/100"
        )
    left, right = scale_arg.split("=", 1)
    return int(left), Path(right)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate observatory scaling summaries across multiple probe-count runs."
    )
    parser.add_argument(
        "--scale",
        action="append",
        required=True,
        help="Scale mapping in the form probe_count=directory",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_scaling",
        help="Directory for aggregated scaling summary outputs",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scale_arg in args.scale:
        probe_count, scale_dir = parse_scale_arg(scale_arg)
        rows.append(summarize_scale(scale_dir, probe_count))

    df = pd.DataFrame(rows).sort_values("probe_count").reset_index(drop=True)
    df.to_csv(outdir / "scaling_summary.csv", index=False)

    print(outdir / "scaling_summary.csv")


if __name__ == "__main__":
    main()
