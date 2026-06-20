# Changelog

All notable changes to Magpie.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-06-20

First cut: the incremental-GP **spine** and **Capability A (active learning)**, built on
AbstractGPs.jl and KernelFunctions.jl, Mooncake-first for automatic differentiation.

### Added

- **`AbstractGPModel <: AbstractGPs.AbstractGP`** — the contract root. GP types that
  support incremental `update`, hyperparameter `fit`, `predmean`, and `predict` on top of
  AbstractGPs' `mean`/`var`/`cov`. Subtyping it inherits `rand`/`logpdf` and the
  Distributions interface for free.
- **`ExactGP`** — exact-regression GP with a cached, incrementally-extended Cholesky
  factor. `update` reuses `AbstractGPs.update_chol` (incremental conditioning equals a
  batch fit to ~1e-9); `nlml` and `fit` optimize the lengthscale (Optimization + LBFGS),
  with the gradient checked against finite differences under Mooncake.
- **`LaplaceGP`** — binary-classification Laplace approximation (Rasmussen & Williams
  Algorithm 3.1, fixed-count unrolled Newton, caches the MAP dual). Differentiates
  cleanly under Mooncake; re-fits on accumulated data each `update`.
- **Acquisitions** — `Straddle` and `RandStraddle` (randomized-width level-set seeking),
  and `BinaryBALD` (calibrated bits-BALD, logistic-corrected Houlsby form, verified
  against numerical quadrature). All are differentiable in the query point.
- **Acquisition maximization** — `acquire(g, a; over)` over an `AcquisitionDomain`: a
  continuous `Box` (grid sweep in low dimension, otherwise Sobol sampling + LBFGS polish)
  or an explicit `Points` set.
- **`ActiveLearner`** — the mutable fit → acquire → observe → update loop (`observe!`,
  `fit!`, `acquire`, `run!`) with history accessors (`posterior_gp`, `queried_points`,
  `all_data`).
- Acceptance exemplars: **A1** (Straddle recovers a level set) and **A2** (BinaryBALD +
  LaplaceGP recover a decision boundary).

### Notes

- Automatic differentiation is **Mooncake-first** via DifferentiationInterface;
  ForwardDiff is used for argmax-over-query-point. Factorizations are dense and routed
  through one internal `_chol` chokepoint (the `Symmetric`-vs-`Matrix` wrapper is
  AD-neutral under Mooncake; see `CLAUDE.md`).
- Not yet built: sparse/inducing GPs, the decoupled (Matheron) sampler, multi-class BALD,
  and the GP-in-SciML (UDE) bridge — the next milestone.

[Unreleased]: https://github.com/jowch/Magpie.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jowch/Magpie.jl/releases/tag/v0.1.0
