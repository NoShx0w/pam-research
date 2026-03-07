# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

import numpy as np

def ols_fit(y, X):
    # y: (n,)
    # X: (n,k) includes intercept if you want
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - sse/sst if sst > 0 else np.nan
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
        "model_B": {"beta": betaB, "R2": r2B}
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
        "entropy_delta_r2": r2_full_B - r2_restr_B
    }

