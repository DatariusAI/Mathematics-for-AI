"""Probability for ML: Bayes' rule and Monte Carlo estimation.

1) A diagnostic-test example of Bayes' rule.
2) A Beta-Binomial update for a click-through rate.
3) A Monte Carlo estimate of pi with its standard error.
Run: python bayes_and_monte_carlo.py
"""
import numpy as np

# 1) Bayes' rule: P(disease | positive)
prior, sensitivity, false_pos = 0.01, 0.95, 0.05
evidence = sensitivity * prior + false_pos * (1 - prior)
print(f"P(disease | positive test) = {sensitivity * prior / evidence:.3f}")

# 2) Beta-Binomial conjugate update
a, b = 1, 1                 # uniform prior on the click rate
clicks, views = 42, 1000
a_post, b_post = a + clicks, b + views - clicks
mean = a_post / (a_post + b_post)
print(f"posterior mean click rate = {mean:.4f}")

# 3) Monte Carlo estimate of pi
rng = np.random.default_rng(2)
n = 200_000
pts = rng.uniform(-1, 1, size=(n, 2))
inside = (pts**2).sum(axis=1) <= 1
est = 4 * inside.mean()
se = 4 * inside.std(ddof=1) / np.sqrt(n)
print(f"pi ~ {est:.4f} +/- {1.96 * se:.4f} (95% interval)")
