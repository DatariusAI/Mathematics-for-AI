"""Differential equations for ML: Euler vs Runge-Kutta 4.

Solves dy/dt = -2y with y(0) = 1 (exact answer e^{-2t}). These solvers
are the building blocks behind neural ODEs and diffusion samplers.
Run: python euler_vs_rk4.py
"""
import numpy as np


def f(t, y):
    return -2 * y


def euler(h, T=2.0):
    y, t = 1.0, 0.0
    while t < T - 1e-12:
        y += h * f(t, y)
        t += h
    return y


def rk4(h, T=2.0):
    y, t = 1.0, 0.0
    while t < T - 1e-12:
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return y


exact = np.exp(-4.0)
for h in [0.2, 0.1, 0.05]:
    print(f"h={h:<5} Euler error={abs(euler(h) - exact):.2e}   RK4 error={abs(rk4(h) - exact):.2e}")
