# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

**v0.1 spine + Capability A on `main`; this branch (`worktree-critical-points`) extends it** — multi-output spine, generic destructure-based `fit`, `LaplaceGP` fitting, the critical-points/transition-state machinery, and reworked worked examples (suite **161** green). What's built:

- **Spine** (`src/spine.jl`, `src/laplace.jl`, `src/fit.jl`): `AbstractGPModel <: AbstractGPs.AbstractGP` is the contract root — it adds `update`/`fit`/`predmean`/`predict` on top of AbstractGPs' `mean`/`var`/`cov`. `ExactGP` — incremental exact-regression GP (cached `(δ, C::Cholesky, α)`; `update` extends the factor via `AbstractGPs.update_chol`); **multi-output** for `d>1` (independent outputs, shared kernel/Cholesky, `n×d` weights, shared variance — aligned with the gp-ude `ExactGPField`; scalar-only paths `fit`/`grad_predict`/`acquire` guard `d>1`). `LaplaceGP` — binary-classification Laplace (unrolled R&W Alg 3.1, caches the MAP dual `a`, Mooncake-clean). All factorizations go through the `_chol` chokepoint.
- **`fit`** is **generic over `AbstractGPModel`**, dispatching on `nlml` via `_fit_xy`/`_recondition` hooks: `ExactGP` (exact log-evidence) and `LaplaceGP` (Laplace log-evidence, R&W 3.32) both fit. It is **destructure-based** (`Optimisers.destructure` over any user-built KernelFunctions kernel) — supports **ARD**, composites (sums/products), per-input-subspace kernels via `SelectTransform`, and a `:auto` lengthscale MAP prior (scalar-ℓ only); a warn-once fires on a scale-less composite. Adding a model = define its `nlml`.
- **Capability A** (`src/acquisitions.jl`, `src/maximize.jl`, `src/loop.jl`, `src/derivatives.jl`, `src/saddle.jl`): acquisitions `Straddle`/`RandStraddle`/`BinaryBALD`/`GradStraddle`/`LocalPenalization`; `acquire(g, a; over)` over a `Box`/`Points` domain; the mutable `ActiveLearner` loop. **Critical points / transition states:** `grad_predict` (AD posterior gradient + Hessian; per-family Matérn prior-gradient-variance — note the Matérn-3/2 constant is `3σ²/ℓ²`), `saddle_walk` (gentlest-ascent), `newton_polish`, `transition_state`, `classify` (Morse index).
- **Tests** (`test/`, 161): self-consistency invariants, Mooncake/finite-difference AD checks, multi-output, ARD/composite kernels, LaplaceGP evidence + fit, and end-to-end exemplars (A1 level-set, A2 BALD boundary, critical points, saddle).
- **Examples** (`examples/`, Literate.jl → docs + CI anti-rot): Müller–Brown single-ended transition state from one minimum (custom anisotropic r²-Taylor Matérn), level-set (ARD ellipse), BALD classification (fitted classifier), volcano DEM (ARD). Author per **[`examples/STYLE.md`](examples/STYLE.md)**.

**Deferred** (documented, not built): SparseGP/inducing, the decoupled (Matheron) sampler, multi-class BALD, **coregionalized/multi-output kernels** (ICM/LMC — see `docs/superpowers/specs/2026-06-23-multi-output-boundary.md`; available via plain AbstractGPs + `MOInput` today), multi-output acquisitions, the generic r²-Taylor derivative kernel (`docs/superpowers/specs/2026-06-23-matern-derivative-variance.md`), and all of **Capability B (GP-in-SciML bridge)** — the next plan.

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
2. **GP-in-SciML bridge** (`docs/research/gp-ude.md`) — GP as ODE field, **through-the-solver**, `GaussAdjoint`+`MooncakeVJP` (outer Mooncake); `SingleShooting` first, with multiple shooting + sparse/inducing + decoupled sampling deferred until forced. ⏳ **next — not started.**
3. *(Then, only if pulled by use:)* linear-constraint kernels, latent-space embedding for high-D dynamics, more likelihoods.

A custom `EnzymeRules` adjoint for the GP solve is a *deferred optimization*, not v1 — Mooncake differentiates the dense path today (the earlier "Enzyme adjoint first" plan was overturned by reconnaissance; see `critique.md`).

**Spikes resolved** (June 2026, `docs/research/spike-results.md`): (1) GP-in-ODE through-solver diff ✅ works (`GaussAdjoint`+`MooncakeVJP`, outer Mooncake; `MooncakeVJP` is unexported but benchmarked ~6× faster than the `ReverseDiffVJP` fallback; `MooncakeAdjoint` buggy); (3) `update_chol` under Mooncake ✅ works (so reuse it). (2) Laplace under Mooncake ✅ resolved — reusing ApproximateGPs' Laplace fails (a `@debug`-macro `try/catch`), but a clean from-scratch Laplace differentiates under Mooncake (Spike 4), so own a ~20-line Laplace and binary BALD is unblocked; (4) GP-UDE cost ⚠️ measured for the worst case (recompute RHS ~GiB at n≈200; ≈n²/≈linear-in-steps) — use the cached-α/inducing design. Remaining empirical work: cost of the cached/inducing GP-UDE field; multiple shooting.

### Pedagogical intent

Understanding the numerics deeply is a motivation, not the deliverable. Keep the **public API clean and performant**; put literate/derivation content in companion notebooks, not in verbose core code.

When writing or reworking the worked examples in `examples/` (Literate.jl tutorials rendered to the docs site and run as CI anti-rot tests), follow **[`examples/STYLE.md`](examples/STYLE.md)** — the example authoring style guide. The load-bearing rule: examples are **problem-driven, not feature-driven**, and **the narrative is structural while the prose stays plain and expository** (collegial, not academic or flowery). It also covers honesty/no-overclaiming, what diagnostics and baselines to show, kernel composition, cross-example coherence, and the Literate/`#src`/`docs/make.jl` mechanics.
