# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from __future__ import annotations

from typing import Tuple

import numpy as np


def smooth(x, W: int = 30) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if W <= 1:
        return x.copy()
    if len(x) < W:
        return np.array([], dtype=float)
    return np.convolve(x, np.ones(W) / W, mode="valid")


def lag_corr(
    x,
    y,
    max_lag: int = 80,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    corr(lag) = corr(x[t], y[t+lag])

    lag > 0: x leads y
    lag < 0: y leads x
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.full_like(lags, fill_value=np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag < 0:
            xs = x[-lag:n]
            ys = y[0:n + lag]
        elif lag > 0:
            xs = x[0:n - lag]
            ys = y[lag:n]
        else:
            xs, ys = x, y

        if len(xs) < 5:
            continue

        if np.std(xs) == 0.0 or np.std(ys) == 0.0:
            corrs[i] = np.nan
        else:
            corrs[i] = np.corrcoef(xs, ys)[0, 1]

    k = int(np.nanargmax(np.abs(corrs)))
    return lags, corrs, int(lags[k]), float(corrs[k])


def align_by_lag(x, y, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns aligned (x_aligned, y_aligned) such that y_aligned[t] = y[t+lag]
    under the same convention as lag_corr.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    if lag < 0:
        return x[-lag:n], y[0:n + lag]
    if lag > 0:
        return x[0:n - lag], y[lag:n]
    return x, y


def ols_fit(y, X):
    # y: (n,)
    # X: (n,k) includes intercept if you want
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return beta, r2, sse


def fit_minimal_models(pi, H):
    Ft, Ft1 = pi[:-1], pi[1:]
    Ht, Ht1 = H[:-1], H[1:]

    XA = np.column_stack([np.ones_like(Ft), Ft, Ht])
    XB = np.column_stack([np.ones_like(Ht), Ht, Ft])

    betaA, r2A, _ = ols_fit(Ft1, XA)
    betaB, r2B, _ = ols_fit(Ht1, XB)

    return {
        "model_A": {"beta": betaA, "R2": r2A},
        "model_B": {"beta": betaB, "R2": r2B},
    }


def granger_delta_r2(pi, H):
    Ft, Ft1 = pi[:-1], pi[1:]
    Ht, Ht1 = H[:-1], H[1:]

    XA_full = np.column_stack([np.ones_like(Ft), Ft, Ht])
    XA_restr = np.column_stack([np.ones_like(Ft), Ft])

    XB_full = np.column_stack([np.ones_like(Ht), Ht, Ft])
    XB_restr = np.column_stack([np.ones_like(Ht), Ht])

    _, r2_full_A, _ = ols_fit(Ft1, XA_full)
    _, r2_restr_A, _ = ols_fit(Ft1, XA_restr)

    _, r2_full_B, _ = ols_fit(Ht1, XB_full)
    _, r2_restr_B, _ = ols_fit(Ht1, XB_restr)

    return {
        "freeze_delta_r2": r2_full_A - r2_restr_A,
        "entropy_delta_r2": r2_full_B - r2_restr_B,
    }
