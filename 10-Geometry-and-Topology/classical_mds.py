"""Geometry for ML: distances, embeddings and classical MDS.

Recovers 2-D coordinates of points from their pairwise distances only,
the idea behind many dimensionality-reduction methods.
Run: python classical_mds.py
"""
import numpy as np

rng = np.random.default_rng(4)
X = rng.uniform(-1, 1, size=(30, 2))
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))   # distance matrix

n = len(D)
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ (D**2) @ J                                   # double centring
vals, vecs = np.linalg.eigh(B)
top = np.argsort(vals)[::-1][:2]
Y = vecs[:, top] * np.sqrt(vals[top])

D_hat = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
print("max distance error after embedding:", f"{np.abs(D - D_hat).max():.2e}")
print("(near zero: the geometry is recovered up to rotation and reflection)")
