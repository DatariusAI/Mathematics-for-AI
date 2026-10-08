"""Information theory for ML: entropy, cross-entropy and KL divergence.

Shows why cross-entropy is the classification loss: it equals the
entropy of the labels plus the KL divergence to the model.
Run: python entropy_cross_entropy_kl.py
"""
import numpy as np


def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def cross_entropy(p, q):
    return -np.sum(p * np.log2(q))


def kl(p, q):
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))


p = np.array([0.7, 0.2, 0.1])   # true distribution
q = np.array([0.5, 0.3, 0.2])   # model's prediction

print(f"H(p)      = {entropy(p):.4f} bits")
print(f"H(p, q)   = {cross_entropy(p, q):.4f} bits")
print(f"KL(p||q)  = {kl(p, q):.4f} bits")
print(f"check: H(p) + KL = {entropy(p) + kl(p, q):.4f} (equals cross-entropy)")
print(f"a fair coin has {entropy(np.array([0.5, 0.5])):.1f} bit of entropy")
