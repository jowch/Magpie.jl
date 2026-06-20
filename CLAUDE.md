# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

`AlphaGP.jl` is a Julia package at the **stub stage** — `src/AlphaGP.jl` is a `greet()` hello-world and nothing has been implemented yet. The real content right now is the thinking, distilled into `docs/research/` (start at `docs/research/README.md`). These are **research notes, not commitments** — no code or architecture has been decided. Read them before writing any non-trivial code; they record not just what to build but why each numerical and autodiff choice was made. `docs/research/critique.md` is the opinionated counterweight (what is worth doing, ranked by payoff) and should be read alongside the synthesis.

## Commands

Standard Julia package workflow (no build step, no lint config in repo):

```julia
# from the repo root
julia --project=.                       # REPL with this package's environment
] instantiate                           # install deps after Project.toml gains [deps]
] test                                  # run the test suite (tests/ does not exist yet)
] add KernelFunctions AbstractGPs       # add a dependency
```

Run the test suite from the shell, or a single test file directly:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. test/runtests.jl                 # whole suite once test/ exists
julia --project=. test/test_cholesky.jl            # a single test file
```

When tests exist, prefer `@testset` blocks in `test/runtests.jl`; run one set by editing `runtests.jl` to `include` only that file, or pass `Pkg.test()` no filter (Julia has no built-in single-`@testset` runner).

## Intended direction

The thesis (see `docs/research/framework-synthesis.md`): **a Julia-ecosystem-native GP package where the bridges are the product** — a GP as a differentiable, uncertainty-aware component that composes with autodiff, SciML, and ML. It is a good, useful, extensible foundation first; SAXS, dynamics discovery, etc. are *example applications*, not architecture drivers.

Locked decisions:

- **Build on, don't reinvent.** Hard-depend on `KernelFunctions.jl` and `AbstractGPs.jl` (reuse its `update_chol`/`update_posterior` incremental plumbing); `DifferentiationInterface.jl` for backend-agnostic AD; `GPLikelihoods.jl` when needed.
- **AD is Mooncake-first**, via DifferentiationInterface. Zygote is a dying baseline (broken on Julia ≥1.12 in KernelFunctions); **Enzyme is a validated-later target** — it has live 2026 correctness bugs on the Cholesky paths we need. Keep factorizations **dense** (sparse Cholesky is broken under both Enzyme and Mooncake). The `Symmetric`-vs-`Matrix` wrapper form is **AD-neutral under Mooncake** — `cholesky` routes to `LAPACK.potrf!` regardless of wrapper, never hitting the ChainRules `Hermitian` rrule bug (#414); AbstractGPs uses `cholesky(Symmetric(...))` throughout (incl. `update_chol`), Spike-3-verified. House style is `cholesky(Symmetric(K); check=false)` via the internal `_chol` chokepoint (`check=false` guards roundoff-induced tiny-negative pivots when `fit` probes extreme lengthscales under Duals). The `Matrix(Symmetric(...))` wrap is needed **only** when validating Enzyme/another ChainRules-consuming backend — re-add then via per-backend dispatch on `_chol` (justified by Enzyme's own bugs #2967/#2964, not #414). See `docs/research/autodiff-frontiers.md` (status box) and `landscape-scan.md` §3.
- **One spine, two capabilities.** The shared primitive is a cached, incrementally-updated GP state (`L`, `α`). On it: (A) a composable **active-learning loop** with the level-set acquisitions (Straddle, BALD-for-classification) that are absent from the whole Julia ecosystem; (B) a **GP-in-SciML bridge** — a GP as an ODE RHS (GP-UDE), through-the-solver, trained via `GaussAdjoint`+`MooncakeVJP` (outer Mooncake; verified feasible). The only Julia prior art, GPDiffEq.jl, is a dormant PoC.
- **Out of scope (anti-sprawl):** comprehensive BO frameworks, EP / neural meta-acquisition, full nonlinear physics kernels (Helfrich — only *linear*-constraint kernels are in scope), GPLVM/GP-attention/quantum. The AL-in-Julia graveyard is the warning.

### Build order

0. **Shared primitives** — the incremental GP state (reuse AbstractGPs' `update_chol`; expose a public "add → update → predict/acquire" contract; state `(X, y, L::LowerTriangular, α)`, `α = K⁻¹y`, mean `kₓ·α`, var `k** − ‖L⁻¹kₓ‖²`) and a native **decoupled (Matheron) sampler** (`DecoupledGPSample` — *not* pure RFF, which suffers variance starvation). Both capabilities depend on these.
1. **Spine + minimal AL loop** — exact regression + binary Laplace classification; acquisitions **Straddle → Randomized Straddle → binary BALD → multi-class BALD** (`docs/research/acquisition-functions.md`). Owner's primary interest, lowest risk.
2. **GP-in-SciML bridge** (`docs/research/gp-ude.md`) — GP as ODE field, **through-the-solver**, sparse+inducing+decoupled sampling, **multiple shooting**, `GaussAdjoint`+`MooncakeVJP` (outer Mooncake); borrow GPDiffEq's derivative-GP + PULL, rebuild training.
3. *(Then, only if pulled by use:)* linear-constraint kernels, latent-space embedding for high-D dynamics, more likelihoods.

A custom `EnzymeRules` adjoint for the GP solve is a *deferred optimization*, not v1 — Mooncake differentiates the dense path today (the earlier "Enzyme adjoint first" plan was overturned by reconnaissance; see `critique.md`).

**Spikes resolved** (June 2026, `docs/research/spike-results.md`): (1) GP-in-ODE through-solver diff ✅ works (`GaussAdjoint`+`MooncakeVJP`, outer Mooncake; `MooncakeVJP` is unexported but benchmarked ~6× faster than the `ReverseDiffVJP` fallback; `MooncakeAdjoint` buggy); (3) `update_chol` under Mooncake ✅ works (so reuse it). (2) Laplace under Mooncake ✅ resolved — reusing ApproximateGPs' Laplace fails (a `@debug`-macro `try/catch`), but a clean from-scratch Laplace differentiates under Mooncake (Spike 4), so own a ~20-line Laplace and binary BALD is unblocked; (4) GP-UDE cost ⚠️ measured for the worst case (recompute RHS ~GiB at n≈200; ≈n²/≈linear-in-steps) — use the cached-α/inducing design. Remaining empirical work: cost of the cached/inducing GP-UDE field; multiple shooting.

### Pedagogical intent

Understanding the numerics deeply is a motivation, not the deliverable. Keep the **public API clean and performant**; put literate/derivation content in companion notebooks, not in verbose core code.
