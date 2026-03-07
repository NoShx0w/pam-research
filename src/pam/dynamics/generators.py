def self_resample_generator(rng, src_texts, n):
    idx = rng.integers(0, len(src_texts), size=n)
    return [src_texts[i] for i in idx]
