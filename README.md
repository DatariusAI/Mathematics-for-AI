# 🧮 Mathematics for AI, ML and Data Science

A free, organised path through the mathematics behind modern AI. Each branch has free books, courses, YouTube lectures, key papers, interactive apps and a runnable code example.

<!-- STAMP:START -->
_Last refreshed: 2026-10-08 14:02 UTC_
<!-- STAMP:END -->

## 🗺️ Branches

| Branch | Why it matters | Runnable example |
|---|---|---|
| [Calculus](01-Calculus/) | Derivatives and the chain rule are how neural networks learn (backpropagation). | [`gradient_descent_from_scratch.py`](01-Calculus/gradient_descent_from_scratch.py) |
| [Linear Algebra](02-Linear-Algebra/) | Data, weights and embeddings are vectors and matrices; PCA, SVD and attention are linear algebra. | [`pca_with_svd.py`](02-Linear-Algebra/pca_with_svd.py) |
| [Probability](03-Probability/) | Models output probabilities; uncertainty, Bayesian methods and generative models rest on probability. | [`bayes_and_monte_carlo.py`](03-Probability/bayes_and_monte_carlo.py) |
| [Statistics](04-Statistics/) | Statistics tells you whether a model is really better, and how sure you can be. | [`bootstrap_and_tests.py`](04-Statistics/bootstrap_and_tests.py) |
| [Discrete Mathematics](05-Discrete-Mathematics/) | Logic, combinatorics and graphs underpin algorithms, knowledge graphs and graph neural networks. | [`graphs_and_pagerank.py`](05-Discrete-Mathematics/graphs_and_pagerank.py) |
| [Optimization](06-Optimization/) | Training a model is solving an optimization problem. | [`optimizers_compared.py`](06-Optimization/optimizers_compared.py) |
| [Information Theory](07-Information-Theory/) | Cross-entropy loss, KL divergence and compression come from information theory. | [`entropy_cross_entropy_kl.py`](07-Information-Theory/entropy_cross_entropy_kl.py) |
| [Numerical Methods](08-Numerical-Methods/) | Floating point, stability and efficient matrix algorithms decide whether training works in practice. | [`stable_softmax.py`](08-Numerical-Methods/stable_softmax.py) |
| [Differential Equations](09-Differential-Equations/) | Neural ODEs, diffusion models and physics-informed networks are built on differential equations. | [`euler_vs_rk4.py`](09-Differential-Equations/euler_vs_rk4.py) |
| [Geometry & Topology](10-Geometry-and-Topology/) | Embeddings, manifold learning and geometric deep learning use geometry and topology. | [`classical_mds.py`](10-Geometry-and-Topology/classical_mds.py) |

## 🧭 Suggested order

1. Linear Algebra and Calculus, in parallel
2. Probability, then Statistics
3. Optimization
4. Information Theory and Discrete Mathematics
5. Numerical Methods, Differential Equations, Geometry & Topology as you need them

## 📄 Latest research on the mathematics of ML
Newest arXiv papers on the theory and mathematics of machine learning. Refreshed daily.

<!-- ARXIV:START -->
_Loading on first refresh._
<!-- ARXIV:END -->

## 💻 Popular open-source math-for-ML code
Most-starred GitHub repositories updated in the last 12 months.

<!-- CODE:START -->
_Loading on first refresh._
<!-- CODE:END -->

## ▶️ Start here: three free video series

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

## 📖 One book to start with

[Mathematics for Machine Learning](https://mml-book.github.io/) by Deisenroth, Faisal and Ong. Free, and covers most branches above.

## ▶️ Run all examples

```bash
pip install numpy
for f in */*.py; do echo "== $f"; python "$f"; done
```

---
Curated by [Mohammad Alrashed](https://github.com/DatariusAI). Part of a series of free AI hubs. Suggestions and corrections welcome by pull request.
