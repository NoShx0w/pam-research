import numpy as np

from pam.tip import InvariantSpec, InvariantPerceptron
from pam.corpora import texts_C, texts_Cp, texts_Cp2, texts_Cp3, texts_Cp4
from pam.types import RunParams
from pam.dynamics import run_quench
from pam.injectors import (
    top_k_signatures,
    mutation_injector_multi_sig_factory,
    mixture_injector_factory,
    self_resample_generator,
)
from pam.metrics.entropy import compute_entropy_series
from pam.metrics.macrostate import sliding_piF, macrostate_from_microstructure
from pam.metrics.lag import smooth, lag_corr
from pam.metrics.regression import granger_delta_r2, fit_minimal_models


CORPORA = {
    "C": texts_C,
    "Cp": texts_Cp,
    "Cp2": texts_Cp2,
    "Cp3": texts_Cp3,
    "Cp4": texts_Cp4,
}


def build_tip():
    invariants = [
        InvariantSpec("reflective", threshold=0.6),
        InvariantSpec("coherent", threshold=0.6),
        InvariantSpec("playful_serious", threshold=0.6),
        InvariantSpec("geometric", threshold=0.6),
    ]
    return InvariantPerceptron(invariants=invariants, mode="heuristic")


def build_injector(tip, texts0, k=2, attempts_per_sample=12):
    targets = top_k_signatures(tip, texts0, k=k)
    anchor_inj = mutation_injector_multi_sig_factory(tip, targets, attempts_per_sample=attempts_per_sample)
    return mixture_injector_factory(anchor_inj, self_resample_generator)


def macro_fn_factory(bd_thresh=0.05, mg_thresh=15.0):
    return lambda corp_snaps, tip_, p: macrostate_from_microstructure(
        corp_snaps, tip_, p,
        window=6, step=1, sample_every=1,
        bd_thresh=bd_thresh, mg_thresh=mg_thresh
    )


def run_one_seed(*, seed: int, texts0, tip, mix_inj, params: RunParams, alpha: float, W: int):
    params_seed = type(params)(
        alpha=params.alpha, r=params.r, seed=seed, iters=params.iters,
        anchor_set_size=params.anchor_set_size,
        store_snapshots=True, store_every=1
    )

    result = run_quench(
        texts0=texts0,
        tip=tip,
        mixture_injector=mix_inj,
        params=params_seed,
        alpha_schedule=lambda it: alpha,
        source_window=10,
        macrostate_fn=macro_fn_factory(),
    )

    corp_snaps = result.corp_snapshots
    states = result.states

    entropy_reports = compute_entropy_series(corp_snaps, tip, anchor_set_size=params_seed.anchor_set_size, sample_every=1)
    H_joint = np.array([r.H_joint for r in entropy_reports], dtype=float)
    K = np.array([r.K for r in entropy_reports], dtype=float)

    F_raw = np.array([1.0 if s == "F" else 0.0 for s in states], dtype=float)
    n = min(len(F_raw), len(H_joint))
    F_raw = F_raw[:n]
    H_raw = H_joint[:n]

    deltas = granger_delta_r2(F_raw, H_raw)

    pi = sliding_piF(states, W=W)
    Hs = smooth(H_joint, W=W)
    m = min(len(pi), len(Hs))
    pi = pi[:m]
    Hs = Hs[:m]

    lags, corrs, best_lag, best_corr = lag_corr(pi, Hs, max_lag=80)
    idx0 = int(np.where(np.array(lags) == 0)[0][0])
    corr0 = float(corrs[idx0])

    return {
        "seed": seed,
        "best_lag": int(best_lag),
        "best_corr": float(best_corr),
        "corr0": corr0,
        "delta_r2_freeze": float(deltas["freeze_delta_r2"]),
        "delta_r2_entropy": float(deltas["entropy_delta_r2"]),
        "var_H": float(np.var(H_joint)),
        "K_min": float(np.min(K)),
        "K_max": float(np.max(K)),
    }


def run_one(*, texts0, tip, mix_inj, params: RunParams, alpha: float, W: int):
    result = run_quench(
        texts0=texts0,
        tip=tip,
        mixture_injector=mix_inj,
        params=params,
        alpha_schedule=lambda it: alpha,
        source_window=10,
        macrostate_fn=macro_fn_factory(),
    )
    corp_snaps = result.corp_snapshots
    states = result.states

    entropy_reports = compute_entropy_series(corp_snaps, tip, anchor_set_size=params.anchor_set_size, sample_every=1)
    H_joint = np.array([r.H_joint for r in entropy_reports], dtype=float)
    H_marg = np.array([r.H_marginal for r in entropy_reports], dtype=float)
    K = np.array([r.K for r in entropy_reports], dtype=float)
    MI = H_marg - H_joint

    F = np.array([1.0 if s == "F" else 0.0 for s in states], dtype=float)
    n = min(len(F), len(H_joint))
    F = F[:n]
    Hj = np.maximum(H_joint[:n], 0.0)
    MI = MI[:n]

    models_H = fit_minimal_models(F, Hj)
    deltas_H = granger_delta_r2(F, Hj)

    models_MI = fit_minimal_models(F, MI)
    deltas_MI = granger_delta_r2(F, MI)

    pi = sliding_piF(states, W=W)
    Hj_sm = smooth(H_joint, W=W)
    m = min(len(pi), len(Hj_sm))
    lags, corrs, best_lag, best_corr = lag_corr(pi[:m], Hj_sm[:m], max_lag=80)

    return {
        "states": states,
        "H_joint": H_joint,
        "H_marginal": H_marg,
        "K": K,
        "MI": MI,
        "F_raw": F,
        "Hj_raw": Hj,
        "models_H": models_H,
        "deltas_H": deltas_H,
        "models_MI": models_MI,
        "deltas_MI": deltas_MI,
        "lag": {"best_lag": int(best_lag), "best_corr": float(best_corr)},
    }
   
def run_one_summary(
    *,
    texts0,
    tip,
    mix_inj,
    params: RunParams,
    alpha: float,
    W: int,
    save_trajectory: bool = True,
    trajectory_path: str | None = None,
):
    result = run_quench(
        texts0=texts0,
        tip=tip,
        mixture_injector=mix_inj,
        params=params,
        alpha_schedule=lambda it: alpha,
        source_window=10,
        macrostate_fn=macro_fn_factory(),
    )

    corp_snaps = result.corp_snapshots
    states = result.states

    entropy_reports = compute_entropy_series(
        corp_snaps,
        tip,
        anchor_set_size=params.anchor_set_size,
        sample_every=1,
    )

    H_joint = np.array([r.H_joint for r in entropy_reports], dtype=float)
    K = np.array([r.K for r in entropy_reports], dtype=float)
    F_raw = np.array([1.0 if s == "F" else 0.0 for s in states], dtype=float)

    n = min(len(F_raw), len(H_joint))
    F_raw = F_raw[:n]
    H_joint = H_joint[:n]
    K = K[:n]

    H_joint = np.maximum(H_joint, 0.0)

    piF_mean = float(np.mean(F_raw))

    tail_frac = 0.2
    tail_n = max(1, int(round(len(F_raw) * tail_frac)))
    piF_tail = float(np.mean(F_raw[-tail_n:]))

    H_joint_mean = float(np.mean(H_joint))
    var_H_joint = float(np.var(H_joint))
    H_min = float(np.min(H_joint))
    H_max = float(np.max(H_joint))

    K_min = float(np.min(K))
    K_max = float(np.max(K))

    deltas = granger_delta_r2(F_raw, H_joint)

    pi = sliding_piF(states, W=W)
    Hj_sm = smooth(H_joint, W=W)

    m = min(len(pi), len(Hj_sm))
    pi = pi[:m]
    Hj_sm = Hj_sm[:m]

    lags, corrs, best_lag, best_corr = lag_corr(pi, Hj_sm, max_lag=80)
    idx0 = int(np.where(np.array(lags) == 0)[0][0])
    corr0 = float(corrs[idx0])

    if save_trajectory and trajectory_path:
        path = Path(trajectory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            F_raw=F_raw,
            H_joint=H_joint,
            K=K,
            pi=pi,
            Hj_sm=Hj_sm,
            lags=np.asarray(lags, dtype=int),
            corrs=np.asarray(corrs, dtype=float),
        )

    return {
        "piF_mean": piF_mean,
        "piF_tail": piF_tail,
        "H_joint_mean": H_joint_mean,
        "var_H_joint": var_H_joint,
        "H_min": H_min,
        "H_max": H_max,
        "K_min": K_min,
        "K_max": K_max,
        "corr0": corr0,
        "delta_r2_freeze": float(deltas["freeze_delta_r2"]),
        "delta_r2_entropy": float(deltas["entropy_delta_r2"]),
        "best_lag": int(best_lag),
        "best_corr": float(best_corr),
    }
