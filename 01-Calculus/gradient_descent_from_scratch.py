"""Calculus for ML: derivatives, the chain rule and gradient descent.

Fits y = w*x + b by gradient descent, with gradients derived by hand
using the chain rule, then checks them against finite differences.
Run: python gradient_descent_from_scratch.py
"""
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 200)
y = 3.0 * x + 0.5 + rng.normal(0, 0.1, 200)


def loss(w, b):
    return np.mean((w * x + b - y) ** 2)


def grads(w, b):
    # dL/dw = mean(2 * (wx + b - y) * x), dL/db = mean(2 * (wx + b - y))
    r = w * x + b - y
    return np.mean(2 * r * x), np.mean(2 * r)


def numeric_grads(w, b, h=1e-6):
    gw = (loss(w + h, b) - loss(w - h, b)) / (2 * h)
    gb = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
    return gw, gb


w, b, lr = 0.0, 0.0, 0.1
print("analytic vs numeric gradient at start:", grads(w, b), numeric_grads(w, b))
for step in range(200):
    gw, gb = grads(w, b)
    w, b = w - lr * gw, b - lr * gb
    if step % 50 == 0:
        print(f"step {step:3d}  loss={loss(w, b):.4f}  w={w:.3f}  b={b:.3f}")
print(f"final: w={w:.3f} (true 3.0), b={b:.3f} (true 0.5)")
