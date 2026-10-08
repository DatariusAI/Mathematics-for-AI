"""Discrete mathematics for ML: graphs, breadth-first search and PageRank.

Builds a small directed graph, finds shortest hop counts with BFS, and
ranks nodes with PageRank via power iteration (a Markov chain).
Run: python graphs_and_pagerank.py
"""
from collections import deque
import numpy as np

edges = {"A": ["B", "C"], "B": ["C"], "C": ["A"], "D": ["C"], "E": ["A", "D"]}
nodes = sorted(set(edges) | {v for vs in edges.values() for v in vs})


def bfs(start):
    dist, q = {start: 0}, deque([start])
    while q:
        u = q.popleft()
        for v in edges.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


print("hops from E:", bfs("E"))

# PageRank
idx = {n: i for i, n in enumerate(nodes)}
N, d = len(nodes), 0.85
M = np.zeros((N, N))
for u, vs in edges.items():
    for v in vs:
        M[idx[v], idx[u]] = 1 / len(vs)
r = np.full(N, 1 / N)
for _ in range(100):
    r = (1 - d) / N + d * M @ r
for n in sorted(nodes, key=lambda n: -r[idx[n]]):
    print(f"PageRank {n}: {r[idx[n]]:.3f}")
