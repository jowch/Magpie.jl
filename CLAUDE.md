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
- **AD is Mooncake-first**, via DifferentiationInterface. Zygote is a dying baseline (broken on Julia ≥1.12 in KernelFunctions); **Enzyme is a validated-later target** — it has live 2026 correctness bugs on the Cholesky paths we need. Keep factorizations **dense** (sparse Cholesky is broken under both Enzyme and Mooncake); wrap symmetric kernel matrices in `Matrix(...)` before `cholesky`. See `docs/research/autodiff-frontiers.md` (status box) and `landscape-scan.md` §3.
- **One spine, two capabilities.** The shared primitive is a cached, incrementally-updated GP state (`L`, `α`). On it: (A) a composable **active-learning loop** with the level-set acquisitions (Straddle, BALD-for-classification) that are absent from the whole Julia ecosystem; (B) a **GP-in-SciML bridge** — a GP as an ODE RHS (GP-UDE), through-the-solver, trained via `GaussAdjoint`+`MooncakeVJP`. The only Julia prior art, GPDiffEq.jl, is a dormant PoC.
- **Out of scope (anti-sprawl):** comprehensive BO frameworks, EP / neural meta-acquisition, full nonlinear physics kernels (Helfrich — only *linear*-constraint kernels are in scope), GPLVM/GP-attention/quantum. The AL-in-Julia graveyard is the warning.

### Build order

1. **Spine + minimal AL loop** — reuse AbstractGPs' `update_chol`; expose a public "add point → update → predict/acquire" contract; add Straddle/BALD; exact regression + Laplace classification first. Lowest risk, the owner's primary interest, and the SciML bridge needs the same primitive. The persistent state is `(X, y, L::LowerTriangular, α)` with `α = K⁻¹y`; predictive mean `kₓ·α`, variance `k** − ‖L⁻¹kₓ‖²`, LML reads off `L`'s diagonal.
2. **GP-in-SciML bridge** — proves the interop thesis; stresses the AD/solver contracts hardest.
3. *(Then, only if pulled by use:)* linear-constraint kernels, inducing-point/sparse representations when profiling demands, more likelihoods.

Note: a custom `EnzymeRules` adjoint for the GP solve is a *deferred optimization*, not v1 — Mooncake differentiates the dense path today (the earlier "Enzyme adjoint first" plan was overturned by reconnaissance; see `critique.md`).

### Pedagogical intent

Understanding the numerics deeply is a motivation, not the deliverable. Keep the **public API clean and performant**; put literate/derivation content in companion notebooks, not in verbose core code.
