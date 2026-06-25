# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

**v0.1 — spine + Capability A on `main`; this branch (`gp-ude`) adds Capability B (the GP-in-SciML bridge) on top of the multi-output spine + critical-points work just merged from main.** What's built:

- **Spine** (`src/spine.jl`, `src/laplace.jl`, `src/fit.jl`): `AbstractGPModel <: AbstractGPs.AbstractGP` is the contract root — it adds `update`/`fit`/`predmean`/`predict` on top of AbstractGPs' `mean`/`var`/`cov`. `ExactGP` — incremental exact-regression GP (cached `(δ, C::Cholesky, α)`; `update` extends the factor via `AbstractGPs.update_chol`); **multi-output** for `d>1` (independent outputs, shared kernel/Cholesky, `n×d` weights, shared variance — aligned with the gp-ude `ExactGPField`; scalar-only paths `fit`/`grad_predict`/`acquire` guard `d>1`). `LaplaceGP` — binary-classification Laplace (unrolled R&W Alg 3.1, caches the MAP dual `a`, Mooncake-clean). All factorizations go through the `_chol` chokepoint.
- **`fit`** is **generic over `AbstractGPModel`**, dispatching on `nlml` via `_fit_xy`/`_recondition` hooks: `ExactGP` (exact log-evidence) and `LaplaceGP` (Laplace log-evidence, R&W 3.32) both fit. It is **destructure-based** (`Optimisers.destructure` over any user-built KernelFunctions kernel) — supports **ARD**, composites (sums/products), per-input-subspace kernels via `SelectTransform`, and a `:auto` lengthscale MAP prior (scalar-ℓ only); a warn-once fires on a scale-less composite. Adding a model = define its `nlml`.
- **Capability A** (`src/acquisitions.jl`, `src/maximize.jl`, `src/loop.jl`, `src/derivatives.jl`, `src/saddle.jl`): acquisitions `Straddle`/`RandStraddle`/`BinaryBALD`/`GradStraddle`/`LocalPenalization`; `acquire(g, a; over)` over a `Box`/`Points` domain; the mutable `ActiveLearner` loop. **Critical points / transition states:** `grad_predict` (AD posterior gradient + Hessian; per-family Matérn prior-gradient-variance — note the Matérn-3/2 constant is `3σ²/ℓ²`), `saddle_walk` (gentlest-ascent), `newton_polish`, `transition_state`, `classify` (Morse index).
- **Capability B — GP-in-SciML bridge** (`src/gpude.jl`, `ext/MagpieSciMLExt.jl`, `src/eval.jl`): a GP as an ODE right-hand-side, trained **through the solver** via `GaussAdjoint`+`MooncakeVJP` (outer Mooncake). `ExactGPField` (exact-regression field) + `SVGPField` (sparse, trained via a sampled reparameterized Matheron ELBO — the decoupled sampler is on the AD training path) + `CompositeField` (known physics + GP residual); both `SingleShooting` and `MultipleShooting`; the ergonomic `train!`/`posterior`/`propagate` API, with `PULL` (Euler moment-matching) and `Pathwise` (decoupled-ensemble) uncertainty propagation. `posterior(field)` reconstructs per-output single-output spine `ExactGP`s (each `d=1`), so it composes with the multi-output spine above. GP-UDE through-solver training is multi-basin / BLAS-sensitive — see the roadmap follow-up below.
- **Tests** (`test/`): self-consistency invariants, Mooncake/finite-difference AD checks, multi-output, ARD/composite kernels, LaplaceGP evidence + fit, and end-to-end exemplars (A1 level-set, A2 BALD boundary, critical points, saddle); the through-solver Capability-B tests (GP-UDE recovery/calibration) are gated behind `MAGPIE_TEST_SCIML` and run green when enabled.
- **Examples** (`examples/`, Literate.jl → docs + CI anti-rot): Müller–Brown single-ended transition state from one minimum (custom anisotropic r²-Taylor Matérn), level-set (ARD ellipse), BALD classification (fitted classifier), volcano DEM (ARD); and the GP-UDE set (Lotka-Volterra, Van der Pol, FitzHugh-Nagumo, scale-forcing, identifiability). Author per **[`examples/STYLE.md`](examples/STYLE.md)**. The GP-UDE anti-rot gates are BLAS-robust API/structural invariants (not exact recovery values) — see the roadmap follow-up.

**Deferred** (documented, not built): multi-class BALD, **coregionalized/multi-output kernels** (ICM/LMC — see `.superpowers/specs/2026-06-23-multi-output-boundary.md`; available via plain AbstractGPs + `MOInput` today), multi-output acquisitions, the generic r²-Taylor derivative kernel (`.superpowers/specs/2026-06-23-matern-derivative-variance.md`), and high-D latent embedding for GP-UDE. (SparseGP/inducing, the decoupled Matheron sampler, and Capability B are now **built** — see above.)

The design rationale lives in `docs/research/` (start at `docs/research/README.md`) — research notes recording not just what to build but *why* each numerical and autodiff choice was made; `docs/research/critique.md` is the opinionated counterweight (effort ranked by payoff). Read them before non-trivial changes. **Note:** `docs/research/` is kept local and is gitignored — it is not part of the public repository, so the `docs/research/...` references throughout this file resolve only in a local working tree.

**Process docs (design specs + implementation plans) belong in `.superpowers/`, not in the repo.** It is gitignored (local-only, like `docs/research/`); write all spec/plan working artifacts there — never under `docs/` — so they stay out of the public surface. The `.superpowers/...` references in this file resolve only in a local working tree.

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
  - **Kernel-agnostic GP-UDE fields (the enabler for the two below).** The GP-UDE fields currently hard-code one kernel form — scaled SE with a single scalar lengthscale — at three layers: the hyper layout (`nhyp = 2 + d`, i.e. exactly `[logℓ, logσ]`), the through-solver RHS (`kk = _kernel(_pf[1], _pf[2])`, rebuilt from two scalars inside the differentiated path), and the Matheron sampler (SE spectral density `ω = randn ./ ℓ`). The field constructors already *accept* a `kernel::Kernel` argument but training ignores it — so the API promises a flexibility it doesn't deliver. The fix is to adopt the spine `fit`'s **`Optimisers.destructure`/`restructure`** substrate: carry a destructured kernel-param block (arbitrary length `nθ`) instead of the fixed two scalars, so ARD / Matérn / composite / linear-constraint kernels all train through the solver. **Gate (spike first):** verify `restructure` differentiates cleanly through `GaussAdjoint`+`MooncakeVJP` — the kernel is rebuilt *inside* the differentiated RHS, and today's two-raw-scalar `_kernel` path sidesteps exactly this. Secondary piece: the Matheron sampler's SE-spectral assumption — PULL is already kernel-agnostic (uses `cov`/`var`); Pathwise needs per-kernel spectral sampling or a clear error for non-SE kernels until generalized. This is the substrate the next two bullets build on.
  - **Linear-constraint kernels** — divergence-free / curl-free / linear-PDE-constrained GPs (in scope; only *linear*-constraint kernels, per anti-sprawl). Builds on kernel-agnostic fields above.
  - **Latent-space embedding** — GP-UDE in a learned latent space for high-dimensional dynamics. Also wants kernel-agnostic fields (ARD / per-subspace kernels).
  - **Multi-class BALD** — the one Capability-A piece deferred (needs AugmentedGPLikelihoods.jl).
- **Tracked follow-ups from the SVGP arc** (non-blocking hygiene): re-derive `examples/gp_ude_scale_forcing.jl` recovery thresholds for the sampled objective (it's a by-hand demo now — multi-trajectory sampled training exceeds the 30-min docs-CI budget); optional nightly/scheduled job running the `MAGPIE_TEST_SCIML_SLOW` calibration file; optionally strengthen the §5.3 estimator-consistency gate to value-equivalence against the analytic mean-field+trace form.
- **GP-UDE training robustness (research follow-up).** Investigation (June 2026, 4-agent + empirical) found the through-solver GP-hyperparameter optimization is **multi-basin and ill-conditioned**: a good well-generalizing basin (large smooth lengthscale) vs a degenerate wiggly-overfit basin (tiny ℓ, σ_obs blows up). Which one the ADAM→LBFGS optimizer lands in is **sensitive to the BLAS build / hardware** (a different OpenBLAS in Julia 1.10 vs 1.12 flips lotka from rmse 0.16 to 4.2) — *not* fixable by pinning deps, pinning BLAS threads, more iterations, or restarts+keep-best (training **loss is misaligned with clean-trajectory generalization** — best-by-loss selects the overfit). This is a recognized property of GP marginal-likelihood + neural/UDE-ODE training (scikit-learn LML multimodality; Summers & Dinneen ICML'21 training instability; OpenBLAS non-associativity). **Partial mitigation shipped:** an opt-in median-heuristic data-driven lengthscale init (`_median_lengthscale`, via `logℓ0 = nothing` on the field constructors) — starts the optimizer on the data scale instead of `ℓ=1`, reducing but not eliminating the fragility (lotka bad-BLAS 4.2 → ≈0.8). (Auto-centering the log-ℓ prior on the init inside `train!` was tried and **reverted**: it silently changes any field built with an explicit per-seed init — e.g. the `identifiability` example, which deliberately keeps `logℓ_ref=0` — so re-center the prior explicitly via `logℓ_ref` where wanted.) CI gates are therefore written as **BLAS-robust invariants** (runs/finite/PSD/valid-fraction + loose sanity bounds, single-threaded BLAS), not exact recovery values. **Real fix (deferred):** a robustly-generalizing objective — stronger/correct regularization, an σ_obs floor (prevent fitting noise as signal), collocation/gradient-matching warm-start (SciML), iterative time-horizon growth, and held-out/CV selection across restarts instead of training-loss selection.

A custom `EnzymeRules` adjoint for the GP solve is a *deferred optimization*, not v1 — Mooncake differentiates the dense path today (the earlier "Enzyme adjoint first" plan was overturned by reconnaissance; see `critique.md`).

**Spikes resolved** (June 2026, `docs/research/spike-results.md`): (1) GP-in-ODE through-solver diff ✅ works (`GaussAdjoint`+`MooncakeVJP`, outer Mooncake; `MooncakeVJP` is unexported but benchmarked ~6× faster than the `ReverseDiffVJP` fallback; `MooncakeAdjoint` buggy); (3) `update_chol` under Mooncake ✅ works (so reuse it). (2) Laplace under Mooncake ✅ resolved — reusing ApproximateGPs' Laplace fails (a `@debug`-macro `try/catch`), but a clean from-scratch Laplace differentiates under Mooncake (Spike 4), so own a ~20-line Laplace and binary BALD is unblocked; (4) GP-UDE cost ⚠️ measured for the worst case (recompute RHS ~GiB at n≈200; ≈n²/≈linear-in-steps) — use the cached-α/inducing design. Remaining empirical work: cost of the cached/inducing GP-UDE field; multiple shooting.

### Pedagogical intent

Understanding the numerics deeply is a motivation, not the deliverable. Keep the **public API clean and performant**; put literate/derivation content in companion notebooks, not in verbose core code.

When writing or reworking the worked examples in `examples/` (Literate.jl tutorials rendered to the docs site and run as CI anti-rot tests), follow **[`examples/STYLE.md`](examples/STYLE.md)** — the example authoring style guide. The load-bearing rule: examples are **problem-driven, not feature-driven**, and **the narrative is structural while the prose stays plain and expository** (collegial, not academic or flowery). It also covers honesty/no-overclaiming, what diagnostics and baselines to show, kernel composition, cross-example coherence, and the Literate/`#src`/`docs/make.jl` mechanics.
