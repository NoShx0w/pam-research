#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


"""
run_refresh_data_for_figures.py

Canonical refresh script for figure-facing data products.

Purpose
-------
Recompute the minimal set of derived CSV outputs used by the paper figures
after new backfill / probe data has landed, without rerunning the entire
geometry pipeline.
"""


ROOT = Path.cwd()


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("==>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def ensure_exists(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n- " + "\n- ".join(missing))


def build_python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    canonical = "./:./src:./experiments"
    env["PYTHONPATH"] = canonical if not existing else f"{canonical}:{existing}"
    return env


def refresh_transition_rate(python_bin: str, env: dict[str, str], *, within_k: int) -> None:
    ensure_exists([ROOT / "outputs/fim_ops_scaled/scaled_probe_paths.csv"])
    run_cmd(
        [
            python_bin,
            "experiments/fim_transition_rate.py",
            "--paths-csv",
            "outputs/fim_ops_scaled/scaled_probe_paths.csv",
            "--outdir",
            "outputs/fim_transition_rate",
            "--within-k",
            str(within_k),
        ],
        env=env,
    )


def refresh_horizon(python_bin: str, env: dict[str, str]) -> None:
    ensure_exists([ROOT / "outputs/fim_ops_scaled/scaled_probe_metrics.csv"])
    run_cmd(
        [
            python_bin,
            "experiments/studies/fim_horizon_from_probes.py",
            "--input-csv",
            "outputs/fim_ops_scaled/scaled_probe_metrics.csv",
            "--outdir",
            "outputs/fim_horizon",
        ],
        env=env,
    )


def refresh_temporal(python_bin: str, env: dict[str, str]) -> None:
    ensure_exists([ROOT / "outputs/fim_ops_scaled/scaled_probe_paths.csv"])
    run_cmd(
        [
            python_bin,
            "experiments/studies/fim_lazarus_temporal.py",
            "--paths-csv",
            "outputs/fim_ops_scaled/scaled_probe_paths.csv",
            "--outdir",
            "outputs/fim_lazarus_temporal",
        ],
        env=env,
    )


def refresh_scaling_summary(
    python_bin: str,
    env: dict[str, str],
    *,
    scales: list[int],
    scales_root: str,
) -> None:
    scale_args: list[str] = []
    for n in scales:
        scale_dir = Path(scales_root) / str(n)
        scale_args.extend(["--scale", f"{n}={scale_dir.as_posix()}"])
    run_cmd(
        [
            python_bin,
            "experiments/studies/fim_scaling_summary.py",
            *scale_args,
            "--outdir",
            "outputs/fim_scaling",
        ],
        env=env,
    )


def validate_outputs() -> None:
    expected = [
        ROOT / "outputs/fim_transition_rate/transition_rate_summary.csv",
        ROOT / "outputs/fim_transition_rate/transition_rate_labeled.csv",
        ROOT / "outputs/fim_horizon/horizon_predictive_summary_from_probes.csv",
        ROOT / "outputs/fim_lazarus_temporal/lazarus_temporal_summary.csv",
        ROOT / "outputs/fim_scaling/scaling_summary.csv",
    ]
    ensure_exists(expected)
    print("==> validation passed")
    for p in expected:
        print("   ok:", p.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh figure-facing derived data products after backfill."
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable or "python",
        help="Python interpreter to use for child experiment scripts.",
    )
    parser.add_argument(
        "--within-k",
        type=int,
        default=2,
        help="Prediction horizon passed to fim_transition_rate.py",
    )
    parser.add_argument(
        "--skip-scaling",
        action="store_true",
        help="Skip fim_scaling_summary.py refresh.",
    )
    parser.add_argument(
        "--scales-root",
        default="outputs/scales",
        help="Root directory containing per-scale outputs.",
    )
    parser.add_argument(
        "--scales",
        default="10,100,1000,10000,100000",
        help="Comma-separated scale list for scaling summary refresh.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Do not validate expected outputs after refresh.",
    )
    args = parser.parse_args()

    env = build_python_env()
    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]

    print("==> Refreshing figure-facing data products")
    refresh_transition_rate(args.python_bin, env, within_k=args.within_k)
    refresh_horizon(args.python_bin, env)
    refresh_temporal(args.python_bin, env)

    if not args.skip_scaling:
        refresh_scaling_summary(
            args.python_bin,
            env,
            scales=scales,
            scales_root=args.scales_root,
        )

    if not args.no_validate:
        validate_outputs()

    print("==> Refresh complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
