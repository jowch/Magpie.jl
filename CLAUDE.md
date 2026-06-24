# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

**v0.1 — spine + Capability A implemented and on `main`** (test suite green, 40 tests). What's built:

- **Spine** (`src/spine.jl`, `src/laplace.jl`, `src/fit.jl`): `AbstractGPModel <: AbstractGPs.AbstractGP` is the contract root — it adds `update`/`fit`/`predmean`/`predict` on top of AbstractGPs' `mean`/`var`/`cov`, so anything the active-learning loop and SciML bridge drive subtypes it. `ExactGP` — incremental exact-regression GP (cached `(δ, C::Cholesky, α)`; `update` extends the factor via `AbstractGPs.update_chol`; `nlml` + `fit` with a Mooncake-checked gradient). `LaplaceGP` — binary-classification Laplace (clean unrolled R&W Alg 3.1, caches the MAP dual `a`, Mooncake-clean; re-fits on accumulated data each `update`). All factorizations go through the `_chol` chokepoint.
- **Capability A** (`src/acquisitions.jl`, `src/maximize.jl`, `src/loop.jl`): acquisitions `Straddle`, `RandStraddle`, `BinaryBALD` (calibrated bits-BALD, verified vs quadrature); `acquire(g, a; over)` over a `Box` or `Points` domain (low-D grid / Sobol-polish); the mutable `ActiveLearner` loop (`observe!`/`fit!`/`acquire`/`run!` + accessors).
- **Tests** (`test/`): self-consistency invariants, Mooncake-vs-finite-difference AD checks, and two end-to-end exemplars — A1 (Straddle level-set recovery) and A2 (BinaryBALD + LaplaceGP boundary).

**Deferred** (not built): SparseGP/inducing, the decoupled (Matheron) sampler, multi-class BALD, and all of **Capability B (GP-in-SciML bridge)** — the next plan.

The design rationale lives in `docs/research/` (start at `docs/research/README.md`) — research notes recording not just what to build but *why* each numerical and autodiff choice was made; `docs/research/critique.md` is the opinionated counterweight (effort ranked by payoff). Read them before non-trivial changes. **Note:** `docs/research/` is kept local and is gitignored — it is not part of the public repository, so the `docs/research/...` references throughout this file resolve only in a local working tree.

## Commands

Standard Julia package workflow (no build step, no lint config in repo):

```julia
# from the repo root
julia --project=.                       # REPL with this package's environment
] instantiate                           # install deps
] test                                  # run the full test suite
] add SomePackage                       # add a dependency
```

Run the suite from the shell, or a single test file directly:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'       # full suite (test/runtests.jl)
julia --project=. test/test_spine.jl               # a single test file
julia --project=. test/test_acquisitions.jl        # acquisitions only
```

`test/runtests.jl` includes each `test_*.jl` under one top-level `@testset`. To run one set, run its file directly (above) or edit `runtests.jl` to `include` only that file — Julia has no built-in single-`@testset` runner. The AD tests (`test_ad.jl`, and the Mooncake checks in `test_laplace.jl`) precompile Mooncake and take ~30s+ each; the full suite is ~2 min.

### Formatting

Code is formatted with [Runic.jl](https://github.com/fredrikekre/Runic.jl) (non-configurable; no style file) and checked in CI via `fredrikekre/runic-action`. Format before committing:

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace   # needs `runic` on PATH (Julia ≥1.12: julia -e 'using Pkg; Pkg.Apps.add("Runic")')
```

## Intended direction

The thesis (see `docs/research/framework-synthesis.md`): **a Julia-ecosystem-native GP package where the bridges are the product** — a GP as a differentiable, uncertainty-aware component that composes with autodiff, SciML, and ML. It is a good, useful, extensible foundation first; SAXS, dynamics discovery, etc. are *example applications*, not architecture drivers.

Locked decisions:

- **Build on, don't reinvent.** Hard-depend on `KernelFunctions.jl` and `AbstractGPs.jl` (reuse its `update_chol`/`update_posterior` incremental plumbing); `DifferentiationInterface.jl` for backend-agnostic AD; `GPLikelihoods.jl` when needed.
- **AD is Mooncake-first**, via DifferentiationInterface. Zygote is a dying baseline (broken on Julia ≥1.12 in KernelFunctions); **Enzyme is a validated-later target** — it has live 2026 correctness bugs on the Cholesky paths we need. Keep factorizations **dense** (sparse Cholesky is broken under both Enzyme and Mooncake). The `Symmetric`-vs-`Matrix` wrapper form is **AD-neutral under Mooncake** — `cholesky` routes to `LAPACK.potrf!` regardless of wrapper, never hitting the ChainRules `Hermitian` rrule bug (#414); AbstractGPs uses `cholesky(Symmetric(...))` throughout (incl. `update_chol`), Spike-3-verified. House style is `cholesky(Symmetric(K); check=false)` via the internal `_chol` chokepoint (`check=false` guards roundoff-induced tiny-negative pivots when `fit` probes extreme lengthscales under Duals). The `Matrix(Symmetric(...))` wrap is needed **only** when validating Enzyme/another ChainRules-consuming backend — re-add then via per-backend dispatch on `_chol` (justified by Enzyme's own bugs #2967/#2964, not #414). See `docs/research/autodiff-frontiers.md` (status box) and `landscape-scan.md` §3.
- **One spine, two capabilities.** The shared primitive is a cached, incrementally-updated GP state (`L`, `α`). On it: (A) a composable **active-learning loop** with the level-set acquisitions (Straddle, BALD-for-classification) that are absent from the whole Julia ecosystem; (B) a **GP-in-SciML bridge** — a GP as an ODE RHS (GP-UDE), through-the-solver, trained via `GaussAdjoint`+`MooncakeVJP` (outer Mooncake; verified feasible). The only Julia prior art, GPDiffEq.jl, is a dormant PoC.
- **Out of scope (anti-sprawl):** comprehensive BO frameworks, EP / neural meta-acquisition, full nonlinear physics kernels (Helfrich — only *linear*-constraint kernels are in scope), GPLVM/GP-attention/quantum. The AL-in-Julia graveyard is the warning.

### Build order

0. **Shared primitives** — the incremental GP state. ✅ **done**: the `AbstractGPModel` contract + `ExactGP`/`LaplaceGP`, `α = K⁻¹y` cached, `update` via `AbstractGPs.update_chol`. The native **decoupled (Matheron) sampler** is **deferred** — v1 acquisitions are analytic and need no function sample; add it when Thompson sampling or the GP-UDE field first needs a callable draw.
1. **Spine + minimal AL loop** — exact regression + binary Laplace classification; acquisitions **Straddle → Randomized Straddle → binary BALD**. ✅ **done** (`docs/research/acquisition-functions.md`). Multi-class BALD **deferred** (needs AugmentedGPLikelihoods.jl).
2. **GP-in-SciML bridge** (`docs/research/gp-ude.md`) — GP as ODE field, **through-the-solver**, `GaussAdjoint`+`MooncakeVJP` (outer Mooncake). ✅ **done** (on the `gp-ude` branch): `ExactGPField` (exact-regression field) + `SVGPField` (sparse, trained via a **sampled reparameterized Matheron ELBO** — the decoupled sampler is now on the AD training path, replacing the earlier mean-field+local-trace surrogate) + `CompositeField` (known physics + GP residual); both `SingleShooting` and `MultipleShooting`; the ergonomic `train!`/`posterior`/`propagate` API. Multi-class BALD and high-D latent embedding remain deferred.
3. *(Then, only if pulled by use:)* linear-constraint kernels, latent-space embedding for high-D dynamics, more likelihoods.

### Next (post–Capability-B)

Capabilities A and B are built and reviewed on the `gp-ude` branch. The deliberate next step is **consolidation, then use-pulled features** — not speculative breadth.

- **Land Capability B on `main`.** Three validated rounds (correctness foundation → ergonomic API → SVGP sampled-ELBO) sit on `gp-ude`; merge them so `main` carries the whole GP-in-SciML bridge before new work.
- **Application showcase (recommended driver).** Per the thesis — *the bridges are the product; SAXS / dynamics-discovery are examples* — build one real end-to-end use. A genuine application stress-tests the API, surfaces gaps, and **tells us which step-3 feature the use actually pulls** (rather than guessing).
- **Candidate step-3 features** (build when an application pulls them):
  - **Linear-constraint kernels** — divergence-free / curl-free / linear-PDE-constrained GPs (in scope; only *linear*-constraint kernels, per anti-sprawl).
  - **Latent-space embedding** — GP-UDE in a learned latent space for high-dimensional dynamics.
  - **Multi-class BALD** — the one Capability-A piece deferred (needs AugmentedGPLikelihoods.jl).
- **Tracked follow-ups from the SVGP arc** (non-blocking hygiene): re-derive `examples/gp_ude_scale_forcing.jl` recovery thresholds for the sampled objective (it's a by-hand demo now — multi-trajectory sampled training exceeds the 30-min docs-CI budget); optional nightly/scheduled job running the `MAGPIE_TEST_SCIML_SLOW` calibration file; optionally strengthen the §5.3 estimator-consistency gate to value-equivalence against the analytic mean-field+trace form.

A custom `EnzymeRules` adjoint for the GP solve is a *deferred optimization*, not v1 — Mooncake differentiates the dense path today (the earlier "Enzyme adjoint first" plan was overturned by reconnaissance; see `critique.md`).

**Spikes resolved** (June 2026, `docs/research/spike-results.md`): (1) GP-in-ODE through-solver diff ✅ works (`GaussAdjoint`+`MooncakeVJP`, outer Mooncake; `MooncakeVJP` is unexported but benchmarked ~6× faster than the `ReverseDiffVJP` fallback; `MooncakeAdjoint` buggy); (3) `update_chol` under Mooncake ✅ works (so reuse it). (2) Laplace under Mooncake ✅ resolved — reusing ApproximateGPs' Laplace fails (a `@debug`-macro `try/catch`), but a clean from-scratch Laplace differentiates under Mooncake (Spike 4), so own a ~20-line Laplace and binary BALD is unblocked; (4) GP-UDE cost ⚠️ measured for the worst case (recompute RHS ~GiB at n≈200; ≈n²/≈linear-in-steps) — use the cached-α/inducing design. Remaining empirical work: cost of the cached/inducing GP-UDE field; multiple shooting.

### Pedagogical intent

Understanding the numerics deeply is a motivation, not the deliverable. Keep the **public API clean and performant**; put literate/derivation content in companion notebooks, not in verbose core code.
