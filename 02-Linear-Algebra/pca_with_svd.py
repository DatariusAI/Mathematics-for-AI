"""Linear algebra for ML: PCA from the singular value decomposition.

Centres the data, takes the SVD, and shows that the top singular
vectors recover the direction of greatest variance.
Run: python pca_with_svd.py
"""
import numpy as np

rng = np.random.default_rng(1)
# 2-D data stretched along the direction (1, 1)
z = rng.normal(size=(500, 2)) * np.array([3.0, 0.5])
R = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
              [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
X = z @ R.T

Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
explained = S**2 / np.sum(S**2)

print("first principal direction:", np.round(Vt[0], 3), "(expected about +/-[0.707, 0.707])")
print("variance explained:", np.round(explained, 3))

# Same answer from the covariance matrix's eigendecomposition
vals, vecs = np.linalg.eigh(np.cov(Xc.T))
print("top eigenvector of covariance:", np.round(vecs[:, -1], 3))

# Project to 1-D and reconstruct
X1 = Xc @ Vt[:1].T @ Vt[:1]
print("reconstruction error with 1 component:", round(float(np.mean((Xc - X1) ** 2)), 4))
