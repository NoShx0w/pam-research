#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NODE_CSV = "outputs/toy_identity_transport_alignment/node_transport_alignment.csv"
OBS_CSV = "outputs/fim_identity_obstruction/identity_obstruction_nodes.csv"
OUTDIR = Path("outputs/obs023_obstruction_proxy_sweep")
SEAM_THRESHOLD = 0.15
TOP_K = 10

PROXIES = [
    "obstruction_mean_abs_holonomy",
    "obstruction_max_abs_holonomy",
    "obstruction_mean_holonomy",
    "obstruction_signed_weighted_holonomy",
    "obstruction_signed_sum_holonomy",
]


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(NODE_CSV)
    obs = pd.read_csv(OBS_CSV)

    merged = nodes.merge(
        obs[["node_id", "r", "alpha"] + [c for c in PROXIES if c in obs.columns]],
        on=["node_id", "r", "alpha"],
        how="left",
    )

    seam_mask = pd.to_numeric(merged["distance_to_seam"], errors="coerce") <= SEAM_THRESHOLD
    top_transport = set(
        merged.sort_values("transport_align_mean_deg", ascending=False)
        .head(TOP_K)["node_id"]
        .tolist()
    )

    rows = []
    for proxy in PROXIES:
        if proxy not in merged.columns:
            continue

        corr = safe_corr(merged["transport_align_mean_deg"], merged[proxy])

        seam_mean = float(pd.to_numeric(merged.loc[seam_mask, proxy], errors="coerce").mean())
        off_mean = float(pd.to_numeric(merged.loc[~seam_mask, proxy], errors="coerce").mean())
        seam_ratio = seam_mean / off_mean if np.isfinite(off_mean) and abs(off_mean) > 1e-12 else np.nan

        top_proxy = set(
            merged.sort_values(proxy, ascending=False)
            .head(TOP_K)["node_id"]
            .tolist()
        )
        overlap = len(top_transport & top_proxy)

        rows.append(
            {
                "proxy": proxy,
                "corr_with_transport_misalignment": corr,
                "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
                "seam_mean": seam_mean,
                "off_mean": off_mean,
                "seam_ratio": seam_ratio,
                "top_k_overlap": overlap,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["abs_corr", "top_k_overlap", "seam_ratio"],
        ascending=[False, False, False],
    )

    out.to_csv(OUTDIR / "obstruction_proxy_sweep.csv", index=False)

    lines = ["=== OBS-023 Obstruction Proxy Sweep ===", ""]
    for _, row in out.iterrows():
        lines.append(
            f"{row['proxy']}: corr={row['corr_with_transport_misalignment']:.4f}, "
            f"seam_ratio={row['seam_ratio']:.4f}, "
            f"top_k_overlap={int(row['top_k_overlap'])}"
        )
    (OUTDIR / "obstruction_proxy_sweep_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(out["proxy"], out["abs_corr"])
    ax.invert_yaxis()
    ax.set_xlabel("|corr with transport misalignment|")
    ax.set_title("OBS-023 obstruction proxy sweep")
    ax.grid(alpha=0.15, axis="x")
    fig.tight_layout()
    fig.savefig(OUTDIR / "obstruction_proxy_sweep.png", dpi=220)
    plt.close(fig)

    print(OUTDIR / "obstruction_proxy_sweep.csv")
    print(OUTDIR / "obstruction_proxy_sweep_summary.txt")
    print(OUTDIR / "obstruction_proxy_sweep.png")


if __name__ == "__main__":
    main()
