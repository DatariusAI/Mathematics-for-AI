"""Numerical methods for ML: floating point and the stable softmax.

A naive softmax overflows for large logits. Subtracting the maximum
(the log-sum-exp trick) gives the same answer without overflow.
Run: python stable_softmax.py
"""
import numpy as np

print("0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3, "| use np.isclose:", np.isclose(0.1 + 0.2, 0.3))
print("float32 machine epsilon:", np.finfo(np.float32).eps)

logits = np.array([1000.0, 1001.0, 1002.0])

with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(logits) / np.sum(np.exp(logits))
print("naive softmax:", naive)


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def logsumexp(z):
    m = np.max(z)
    return m + np.log(np.sum(np.exp(z - m)))


print("stable softmax:", np.round(softmax(logits), 4))
print("log-sum-exp:", round(float(logsumexp(logits)), 4))
