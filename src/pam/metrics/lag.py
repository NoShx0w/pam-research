# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

import numpy as np
from typing import Tuple

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
            ys = y[0:n+lag]
        elif lag > 0:
            xs = x[0:n-lag]
            ys = y[lag:n]
        else:
            xs, ys = x, y

        if len(xs) < 5:
            continue

        # handle constant arrays safely
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
    x = x[:n]; y = y[:n]

    if lag < 0:
        return x[-lag:n], y[0:n+lag]
    if lag > 0:
        return x[0:n-lag], y[lag:n]
    return x, y
