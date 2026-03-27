
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Sequence


__all__ = [
    "DEFAULT_OBSERVABLES",
    "GROUP_COLS",
    "run_fisher_metric",
]


"""Canonical Fisher-metric stage for the PAM geometry pipeline."""


DEFAULT_OBSERVABLES = ["piF_tail", "H_joint_mean"]
GROUP_COLS = ["corpus", "r", "alpha"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate a Fisher-type metric on the PAM (r, alpha) control manifold."
    )
    parser.add_argument(
        "--index-csv",
        default="outputs/index.csv",
        help="Path to outputs/index.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim",
        help="Directory for FIM outputs",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional corpus filter, e.g. C",
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        default=DEFAULT_OBSERVABLES,
        help="Observable columns used to induce the metric",
    )
    parser.add_argument(
        "--ridge-eps",
        type=float,
        default=1e-8,
        help="Regularization added to residual covariance before inversion",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def estimate_noise_covariance(df: pd.DataFrame, observables, ridge_eps: float):
    """
    Estimate the observable noise covariance from within-cell residuals across seeds.
    Falls back gracefully when replication is sparse.
    """
    grp = df.groupby(GROUP_COLS, dropna=False)
    cell_means = grp[observables].transform("mean")
    residuals = (df[observables] - cell_means).dropna()

    if len(residuals) >= 2:
        sigma = np.cov(residuals.to_numpy().T, ddof=1)
    else:
        sigma = np.diag(np.nanvar(df[observables].to_numpy(), axis=0) + ridge_eps)

    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    if sigma.shape == ():
        sigma = np.array([[float(sigma)]], dtype=float)

    if sigma.shape[0] != len(observables):
        sigma = np.diag(np.nanvar(df[observables].to_numpy(), axis=0) + ridge_eps)

    sigma = sigma + ridge_eps * np.eye(len(observables))
    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        sigma = sigma + (10.0 * ridge_eps) * np.eye(len(observables))
        sigma_inv = np.linalg.pinv(sigma)

    return sigma, sigma_inv


def aggregate_surface(df: pd.DataFrame, observables):
    agg_map = {c: ["mean", "std", "count"] for c in observables}
    agg = df.groupby(GROUP_COLS, dropna=False).agg(agg_map)
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    return agg


def pivot_surface(agg: pd.DataFrame, value_col: str):
    p = agg.pivot_table(index="r", columns="alpha", values=value_col, aggfunc="mean")
    return p.sort_index().sort_index(axis=1)


def finite_difference(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """
    Compute partial derivatives wrt r (rows/xs) and alpha (cols/ys) using
    central differences where possible and one-sided differences at boundaries.
    Missing cells (NaN) propagate naturally.
    """
    nr, na = arr.shape
    dr = np.full_like(arr, np.nan, dtype=float)
    da = np.full_like(arr, np.nan, dtype=float)

    for i in range(nr):
        for j in range(na):
            if not np.isfinite(arr[i, j]):
                continue

            # derivative wrt r
            left_ok = i - 1 >= 0 and np.isfinite(arr[i - 1, j])
            right_ok = i + 1 < nr and np.isfinite(arr[i + 1, j])

            if left_ok and right_ok:
                dr[i, j] = (arr[i + 1, j] - arr[i - 1, j]) / (xs[i + 1] - xs[i - 1])
            elif right_ok:
                dr[i, j] = (arr[i + 1, j] - arr[i, j]) / (xs[i + 1] - xs[i])
            elif left_ok:
                dr[i, j] = (arr[i, j] - arr[i - 1, j]) / (xs[i] - xs[i - 1])

            # derivative wrt alpha
            down_ok = j - 1 >= 0 and np.isfinite(arr[i, j - 1])
            up_ok = j + 1 < na and np.isfinite(arr[i, j + 1])

            if down_ok and up_ok:
                da[i, j] = (arr[i, j + 1] - arr[i, j - 1]) / (ys[j + 1] - ys[j - 1])
            elif up_ok:
                da[i, j] = (arr[i, j + 1] - arr[i, j]) / (ys[j + 1] - ys[j])
            elif down_ok:
                da[i, j] = (arr[i, j] - arr[i, j - 1]) / (ys[j] - ys[j - 1])

    return dr, da


def render_heatmap(data: pd.DataFrame, title: str, outpath: Path, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(data.values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"{x:.2f}" for x in data.index])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def render_principal_direction(theta_df: pd.DataFrame, valid_df: pd.DataFrame, title: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = theta_df.values
    valid = valid_df.values.astype(bool)

    im = ax.imshow(
        np.where(valid, data, np.nan),
        aspect="auto",
        origin="lower",
        cmap="twilight",
        vmin=-np.pi / 2,
        vmax=np.pi / 2,
    )
    ax.set_xticks(range(len(theta_df.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in theta_df.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(theta_df.index)))
    ax.set_yticklabels([f"{x:.2f}" for x in theta_df.index])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="angle (radians)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_fisher_metric(
    index_csv: str | Path,
    outdir: str | Path,
    corpus: str | None = None,
    observables: Sequence[str] | None = None,
    ridge_eps: float = 1e-8,
):
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    required = GROUP_COLS + ["seed"] + list(args.observables)
    ensure_columns(df, required)

    if args.corpus is not None:
        df = df[df["corpus"] == args.corpus].copy()

    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No valid rows remain after filtering.")

    sigma, sigma_inv = estimate_noise_covariance(df, args.observables, args.ridge_eps)
    agg = aggregate_surface(df, args.observables)

    r_values = np.sort(agg["r"].unique())
    a_values = np.sort(agg["alpha"].unique())

    mean_grids = {}
    dr_grids = {}
    da_grids = {}

    if observables is None:
        observables = DEFAULT_OBSERVABLES

    for obs in args.observables:
        p = pivot_surface(agg, f"{obs}_mean")
        p = p.reindex(index=r_values, columns=a_values)
        mean_grids[obs] = p
        dr, da = finite_difference(p.to_numpy(dtype=float), r_values, a_values)
        dr_grids[obs] = dr
        da_grids[obs] = da

    nr = len(r_values)
    na = len(a_values)

    g_rr = np.full((nr, na), np.nan, dtype=float)
    g_ra = np.full((nr, na), np.nan, dtype=float)
    g_aa = np.full((nr, na), np.nan, dtype=float)
    eig1 = np.full((nr, na), np.nan, dtype=float)
    eig2 = np.full((nr, na), np.nan, dtype=float)
    theta = np.full((nr, na), np.nan, dtype=float)
    detg = np.full((nr, na), np.nan, dtype=float)
    traceg = np.full((nr, na), np.nan, dtype=float)
    condg = np.full((nr, na), np.nan, dtype=float)
    valid = np.zeros((nr, na), dtype=bool)

    for i in range(nr):
        for j in range(na):
            grad_r = np.array([dr_grids[obs][i, j] for obs in args.observables], dtype=float)
            grad_a = np.array([da_grids[obs][i, j] for obs in args.observables], dtype=float)

            if not (np.all(np.isfinite(grad_r)) and np.all(np.isfinite(grad_a))):
                continue

            grr = float(grad_r.T @ sigma_inv @ grad_r)
            gra = float(grad_r.T @ sigma_inv @ grad_a)
            gaa = float(grad_a.T @ sigma_inv @ grad_a)
            G = np.array([[grr, gra], [gra, gaa]], dtype=float)

            evals, evecs = np.linalg.eigh(G)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]

            lam1 = float(evals[0])
            lam2 = float(evals[1])
            v1 = evecs[:, 0]

            g_rr[i, j] = grr
            g_ra[i, j] = gra
            g_aa[i, j] = gaa
            eig1[i, j] = lam1
            eig2[i, j] = lam2
            detg[i, j] = float(np.linalg.det(G))
            traceg[i, j] = float(np.trace(G))
            condg[i, j] = np.inf if lam2 <= 0 else lam1 / lam2
            theta[i, j] = float(np.arctan2(v1[1], v1[0]))
            valid[i, j] = True

    base = pd.DataFrame(
        [(r, a) for r in r_values for a in a_values],
        columns=["r", "alpha"],
    )

    metrics = {
        "fim_rr": g_rr,
        "fim_ra": g_ra,
        "fim_aa": g_aa,
        "fim_det": detg,
        "fim_trace": traceg,
        "fim_eig1": eig1,
        "fim_eig2": eig2,
        "fim_cond": condg,
        "fim_theta": theta,
        "fim_valid": valid.astype(int),
    }

    out_df = base.copy()
    for name, arr in metrics.items():
        out_df[name] = arr.reshape(-1)

    for obs in args.observables:
        out_df[f"{obs}_mean"] = mean_grids[obs].to_numpy(dtype=float).reshape(-1)

    out_df["n_seeds"] = (
        agg.pivot_table(index="r", columns="alpha", values=f"{args.observables[0]}_count", aggfunc="mean")
        .reindex(index=r_values, columns=a_values)
        .to_numpy(dtype=float)
        .reshape(-1)
    )

    out_csv = outdir / "fim_surface.csv"
    out_df.to_csv(out_csv, index=False)

    meta_path = outdir / "fim_metadata.txt"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write("PAM Fisher-type metric estimation\n")
        f.write(f"index_csv={args.index_csv}\n")
        f.write(f"corpus={args.corpus}\n")
        f.write(f"observables={args.observables}\n")
        f.write("sigma=\n")
        f.write(np.array2string(sigma, precision=6, suppress_small=False))
        f.write("\n")
        f.write("sigma_inv=\n")
        f.write(np.array2string(sigma_inv, precision=6, suppress_small=False))
        f.write("\n")

    def to_df(arr):
        return pd.DataFrame(arr, index=r_values, columns=a_values)

    render_heatmap(to_df(g_rr), "FIM g_rr", outdir / "fim_rr.png")
    render_heatmap(to_df(g_ra), "FIM g_rα", outdir / "fim_ra.png")
    render_heatmap(to_df(g_aa), "FIM g_αα", outdir / "fim_aa.png")
    render_heatmap(to_df(traceg), "trace(G)", outdir / "fim_trace.png")
    render_heatmap(to_df(detg), "det(G)", outdir / "fim_det.png")
    render_heatmap(to_df(np.log10(np.clip(detg, 1e-16, None))), "log10 det(G)", outdir / "fim_log10_det.png")
    render_heatmap(to_df(eig1), "largest eigenvalue", outdir / "fim_eig1.png")
    render_heatmap(to_df(eig2), "smallest eigenvalue", outdir / "fim_eig2.png")
    render_heatmap(
        to_df(np.log10(np.clip(condg, 1.0, None))),
        "log10 condition number",
        outdir / "fim_log10_cond.png",
    )
    render_principal_direction(
        to_df(theta),
        to_df(valid.astype(float)),
        "principal sensitivity direction",
        outdir / "fim_theta.png",
    )

    print(f"Wrote {out_csv}")
    print(f"Wrote {meta_path}")
    print(f"Plots saved under {outdir}")

    return out_df


def main():
    return run_fisher_metric()


if __name__ == "__main__":
    main()

