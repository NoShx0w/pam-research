def progress_bar(i: int, n: int, *, width: int = 30, prefix: str = "") -> str:
    """
    Returns a string progress bar for 1-indexed i in [1..n].
    """
    i = max(1, min(i, n))
    frac = i / n
    filled = int(round(width * frac))
    bar = "█" * filled + "░" * (width - filled)
    pct = int(round(100 * frac))
    return f"{prefix}[{bar}] {pct:3d}% ({i}/{n})"