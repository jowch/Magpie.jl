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

## Intended architecture

The project is **a GP-native sequential experimental design (active learning / DoE) library**, built from first principles with autodiff as a first-class concern — not another general GP package. It builds on `KernelFunctions.jl` (and `AbstractGPs.jl` as an API/interop target — see `critique.md` on why the hot loop may not sit on AbstractGPs) and fills the gap above them, which the Julia ecosystem genuinely lacks. The motivating application is SAXS phase-diagram mapping (level-set / boundary estimation, not scalar optimization).

Four layers, built bottom-up (see `docs/research/framework-synthesis.md`):

1. **Numerical foundation** — incremental Cholesky rank-1 update (`O(n²)` per added point) as the persistent state primitive; a custom `EnzymeRules` adjoint for the GP solve via the implicit function theorem (do **not** autodiff through the solver internals — register the mathematically correct gradient); pathwise posterior sampling (Matheron's rule + random Fourier features) as the non-Gaussian inference engine; `DifferentiationInterface.jl` so Enzyme/Mooncake/Zygote are swappable.
2. **Active learning loop** — acquisition functions (EI/LogEI, UCB, BALD-for-classification, IMSPE/Straddle) as differentiable functions of the GP posterior; implicit-diff adjoint through the acquisition argmax; Sobol/LHS cold start; learnable acquisition parameters trained via meta-gradients through the loop.
3. **Physics integration** — linear-operator kernels for physics-informed constraints (Helfrich for membranes, kept general); probabilistic ICs into SciML's `EnsembleProblem`; GP→PDE→acquisition feedback loop.
4. **Representation** — GPLVM encoder/decoder with a deep (Lux) kernel in learned feature space.

### Build-first guidance

The synthesis (`framework-synthesis.md`) names **Layer 1's incremental Cholesky + custom Enzyme adjoint** as the first thing to build. **Note the dissent in `critique.md`:** for the expensive-oracle SAXS regime `n` stays small (tens–hundreds), so the incremental update is a pedagogical exercise, not a performance need — the critique argues for building exact GP regression + Laplace classification + level-set acquisitions (Straddle/BALD) first, since the level-set reframing is the genuinely novel, high-payoff piece. The persistent GP state is `(X, y, L::LowerTriangular, α)` where `α = K⁻¹y`; predictive mean is `kₓ·α`, variance is `k** − ‖L⁻¹kₓ‖²`, log-marginal-likelihood reads off `L`'s diagonal. See `docs/research/sequential-gp-updates.md` (jitter/downdate caveats) and `docs/research/pathwise-sampling.md` (Julia implementation).

### Pedagogical intent

This is a learning tool as much as a work tool. The resolution (see `docs/research/project-scope.md`): keep the **outer API clean and performant**, and put the literate/derivation content in companion notebooks (e.g. deriving the IFT adjoint from scratch), **not** by making the core code verbose. Don't sacrifice the public API's clarity for tutorial annotations.
