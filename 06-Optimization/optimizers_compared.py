"""Optimization for ML: gradient descent, momentum and Adam on a hard valley.

Minimises the Rosenbrock function f(x, y) = (1 - x)^2 + 100 (y - x^2)^2,
whose minimum is at (1, 1), and compares how close each optimizer gets.
Run: python optimizers_compared.py
"""
import numpy as np


def f(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


def run(name, steps=5000):
    p, v, m, s = np.array([-1.5, 2.0]), np.zeros(2), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(p)
        if name == "gd":
            p = p - 1e-3 * g
        elif name == "momentum":
            v = 0.9 * v + g
            p = p - 1e-4 * v
        else:  # adam
            m = 0.9 * m + 0.1 * g
            s = 0.999 * s + 0.001 * g**2
            mh, sh = m / (1 - 0.9**t), s / (1 - 0.999**t)
            p = p - 2e-2 * mh / (np.sqrt(sh) + 1e-8)
    return p


for name in ["gd", "momentum", "adam"]:
    p = run(name)
    print(f"{name:9s} -> point {np.round(p, 3)}, f = {f(p):.5f}")
