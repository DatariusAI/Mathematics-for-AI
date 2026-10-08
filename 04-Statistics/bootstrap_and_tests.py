"""Statistics for ML: bootstrap confidence intervals and a permutation test.

Compares two model versions' accuracies on the same test cases without
assuming any distribution.
Run: python bootstrap_and_tests.py
"""
import numpy as np

rng = np.random.default_rng(3)
model_a = rng.binomial(1, 0.80, 500)   # 1 = correct prediction
model_b = rng.binomial(1, 0.84, 500)

# Bootstrap 95% CI for the accuracy difference
diffs = []
for _ in range(5000):
    idx = rng.integers(0, 500, 500)
    diffs.append(model_b[idx].mean() - model_a[idx].mean())
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"accuracy A={model_a.mean():.3f}  B={model_b.mean():.3f}")
print(f"bootstrap 95% CI for B - A: [{lo:.3f}, {hi:.3f}]")

# Permutation test: is the difference bigger than chance?
observed = model_b.mean() - model_a.mean()
pooled = np.concatenate([model_a, model_b])
count = 0
for _ in range(5000):
    rng.shuffle(pooled)
    if pooled[500:].mean() - pooled[:500].mean() >= observed:
        count += 1
print(f"one-sided permutation p-value = {count / 5000:.4f}")
