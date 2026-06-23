# LaplaceGP Hyperparameter Fitting (generic `fit` over `nlml`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `LaplaceGP` real hyperparameter fitting (replacing the `fit(g::LaplaceGP)=g` no-op) by adding the Laplace log-evidence as `nlml(::LaplaceGP)` and generalizing the existing destructure-based `fit` to dispatch on `nlml`, so any `AbstractGPModel` fits by defining its own evidence.

**Architecture:** Today `fit(g::ExactGP)` owns the destructure → log-space LBFGS → rebuild loop and calls `nlml(g::ExactGP)`. Refactor that loop to `fit(g::AbstractGPModel)`, with three tiny per-model hooks — `nlml(g)`, `_fit_xy(g)` (the training inputs/targets), and `_recondition(g, kernel, X, y)` (rebuild + condition) — so `ExactGP` keeps its exact behavior and `LaplaceGP` gains fitting via its new Laplace-evidence `nlml`. AD-feasibility (Mooncake + ForwardDiff, both matching finite differences, with an interior optimum) was spike-verified 2026-06-23.

**Tech Stack:** Julia ≥1.10, KernelFunctions.jl, AbstractGPs.jl, Optimization.jl + OptimizationOptimJL (LBFGS), DifferentiationInterface.jl (ForwardDiff/Mooncake), Optimisers.jl (`destructure`).

## Global Constraints

- **`ExactGP` `fit` stays byte-for-byte equivalent** — the auto-wrap (`1.0*k`), all-positive-leaf validation, the `:auto` lengthscale MAP prior via `_lengthscale`, the metric-based `_default_ad`, and the `d>1` guard are all preserved. The full existing suite (153 tests) stays green; the A1/A2/critpoints exemplars are unchanged.
- **`nlml(::LaplaceGP)` is the Laplace log-evidence** (R&W eq. 3.32): `½(f̂−m)ᵀa − log p(y|f̂) + Σ log Lᵢᵢ`, computed from the cached `a`/`W`/`L` and `f̂ = K·a + m`. Spike-verified to differentiate under Mooncake and ForwardDiff (both match finite differences).
- **Single-output only** — `fit` guards `_outputdim(g) == 1` (preserves the `ExactGP` `d>1` guard; `LaplaceGP` is single-output, `_outputdim` defaults to 1).
- **Julia ≥1.10**; format with Runic before each commit; run the full suite before each commit.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/laplace.jl` | add `nlml(::LaplaceGP)` (Laplace log-evidence); remove the `fit` no-op | 1, 2 |
| `src/fit.jl` | generalize `fit` to `AbstractGPModel`; add `_fit_xy`/`_recondition` hooks | 2 |
| `src/loop.jl` | (no code change; verify `refit_every` now refits `LaplaceGP`) | 2 |
| `test/test_laplace.jl` | `nlml` AD test + evidence-optimum test; `fit` recovery test | 1, 2 |

---

### Task 1: `nlml(::LaplaceGP)` — the Laplace log-evidence

**Files:**
- Modify: `src/laplace.jl` — add `nlml(g::LaplaceGP)` (near the other `LaplaceGP` methods, before the `fit` no-op at line 122)
- Modify: `test/test_laplace.jl` — add the AD + optimum tests

**Interfaces:**
- Consumes: `LaplaceGP` cached fields `a`, `W`, `L`, `x`, `y`, `prior` (existing); `_hasdata(g::LaplaceGP)` (existing).
- Produces: `nlml(g::LaplaceGP) -> Real` (the negative Laplace log-evidence), differentiable w.r.t. kernel hyperparameters under Mooncake and ForwardDiff.

- [ ] **Step 1: Write the failing test** — append to `test/test_laplace.jl`:

```julia
@testset "LaplaceGP nlml (Laplace log-evidence) differentiates and has an interior optimum" begin
    using Magpie: LaplaceGP, nlml, update
    import DifferentiationInterface as DI
    using DifferentiationInterface: AutoForwardDiff, AutoMooncake
    Random.seed!(1)
    X = [randn(2) for _ in 1:25]
    yb = [(x[1] + 0.5x[2] > 0) for x in X]
    loss(logℓ) = nlml(update(LaplaceGP(with_lengthscale(SqExponentialKernel(), exp(only(logℓ)))), X, yb))
    logℓ0 = [log(0.7)]
    gmc = DI.gradient(loss, AutoMooncake(; config = nothing), logℓ0)
    gfd = DI.gradient(loss, AutoForwardDiff(), logℓ0)
    h = 1.0e-6
    gnum = (loss(logℓ0 .+ h) - loss(logℓ0 .- h)) / 2h
    @test all(isfinite, gmc) && isapprox(gmc[1], gnum; rtol = 1.0e-3)   # Mooncake matches FD
    @test all(isfinite, gfd) && isapprox(gfd[1], gnum; rtol = 1.0e-3)   # ForwardDiff matches FD
    # the evidence has an interior minimum over a sensible lengthscale grid (not pinned to a bound)
    grid = log.(0.1:0.1:3.0)
    ℓbest = exp(grid[argmin([loss([lg]) for lg in grid])])
    @test 0.1 < ℓbest < 3.0
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_laplace.jl`
Expected: FAIL — `nlml` has no `LaplaceGP` method (`MethodError`).

- [ ] **Step 3: Implement `nlml(::LaplaceGP)`** — in `src/laplace.jl`, add before the `fit` no-op (≈line 122):

```julia
@doc raw"""
    nlml(g::LaplaceGP) -> Real

Negative Laplace log marginal likelihood (Rasmussen & Williams, eq. 3.32) — the objective
[`fit`](@ref) minimizes for a `LaplaceGP`:

```math
-\log q(y \mid X) = \tfrac{1}{2}(\hat f - m)^\top a \;-\; \log p(y \mid \hat f)\;+\;\sum_i \log L_{ii},
```

where ``\hat f = K a + m`` is the MAP latent, ``a`` the cached dual, and ``L`` the Cholesky factor
of ``B = I + W^{1/2} K W^{1/2}``. Returns `0.0` for an unconditioned classifier.
"""
function nlml(g::LaplaceGP)
    _hasdata(g) || return 0.0
    m = AbstractGPs.mean(g.prior, g.x)
    K = Matrix(Symmetric(AbstractGPs.cov(g.prior, g.x))) + 1.0e-9I
    Ka = K * g.a
    fhat = Ka .+ m
    t = float.(g.y)
    softplus(z) = log1p(exp(-abs(z))) + max(z, zero(z))   # numerically stable log(1 + eᶻ)
    loglik = sum(t .* fhat .- softplus.(fhat))            # logistic log p(y | f̂)
    quad = 0.5 * dot(Ka, g.a)                             # ½(f̂-m)ᵀ K⁻¹ (f̂-m) = ½(Ka)ᵀa
    logdetB = sum(log, diag(g.L))                         # ½ log|B|
    return quad - loglik + logdetB
end
```

(`nlml` is already imported/exported via `src/fit.jl`/`Magpie.jl`; this is an added method on the same generic function. `LinearAlgebra`'s `dot`/`diag`/`Symmetric`/`I` are in scope in the module.)

- [ ] **Step 4: Run the test, then the full suite**

Run: `julia --project=. test/test_laplace.jl` → Expected: PASS.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green (additive method; nothing else changes).

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add src/laplace.jl test/test_laplace.jl
git commit -m "feat(laplace): nlml = Laplace log-evidence (R&W 3.32), AD-differentiable"
```

---

### Task 2: Generalize `fit` over `AbstractGPModel` so `LaplaceGP` fits

**Files:**
- Modify: `src/fit.jl` — change `fit(g::ExactGP; …)` to `fit(g::AbstractGPModel; …)`; add `_fit_xy`/`_recondition` hooks
- Modify: `src/laplace.jl` — delete the `fit(g::LaplaceGP)=g` no-op (line 122-123); add `_fit_xy`/`_recondition` for `LaplaceGP`
- Modify: `test/test_laplace.jl` — add the `fit` recovery test

**Interfaces:**
- Consumes: `nlml(::LaplaceGP)` (Task 1); `destructure`, `_default_ad`, `_lengthscale`, `_outputdim` (existing); `update(::LaplaceGP, X, y)` (existing).
- Produces: `fit(g::AbstractGPModel; restarts=1, ad=nothing, ℓ_prior=:auto)`; `_fit_xy(g) -> (X, y)`; `_recondition(g, kernel, X, y) -> model`. `fit(::LaplaceGP)` now optimizes hyperparameters; `ExactGP` behavior unchanged.

- [ ] **Step 1: Write the failing test** — append to `test/test_laplace.jl`:

```julia
@testset "fit(::LaplaceGP) recovers a sensible lengthscale and improves the evidence" begin
    using Magpie: LaplaceGP, nlml, fit, update
    Random.seed!(2)
    # smooth boundary; a too-short initial lengthscale over-fits → fit should lengthen it
    X = [4.0 .* rand(2) .- 2.0 for _ in 1:50]
    yb = [(x[1] + 0.7x[2] > 0) for x in X]
    g0 = update(LaplaceGP(with_lengthscale(SqExponentialKernel(), 0.15)), X, yb)
    g = fit(g0; restarts = 2)
    @test nlml(g) ≤ nlml(g0) + 1.0e-6                      # evidence improved (or matched)
    @test Magpie._lengthscale(g.prior.kernel) > Magpie._lengthscale(g0.prior.kernel)  # lengthscale grew
    @test g isa LaplaceGP                                  # still a classifier
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_laplace.jl`
Expected: FAIL — `fit(::LaplaceGP)` currently returns `g0` unchanged, so `nlml(g) == nlml(g0)` but the lengthscale is unchanged (`>` fails).

- [ ] **Step 3: Add the per-model hooks** — in `src/laplace.jl`, replace the no-op (lines 122-123):

```julia
# v1: the Laplace path does no hyperparameter refit, so `fit` returns the GP unchanged.
fit(g::LaplaceGP; kwargs...) = g
```

with the fit hooks (the generic `fit` lives in `src/fit.jl`, Step 4):

```julia
# Hooks that let the generic `fit` (src/fit.jl) drive a LaplaceGP: its training data and how to
# rebuild+condition it from a trial kernel. Hyperparameter fitting maximizes the Laplace evidence.
_fit_xy(g::LaplaceGP) = (g.x, g.y)
_recondition(g::LaplaceGP, kernel, X, y) = update(LaplaceGP(kernel; mean = g.prior.mean), X, y)
```

- [ ] **Step 4: Generalize `fit`** — in `src/fit.jl`, add the `ExactGP` hooks just above `fit`, and change the `fit` signature from `ExactGP` to `AbstractGPModel`, replacing the data-extraction and reconditioning lines with the hooks. The full rewritten `fit` (replacing the current `function fit(g::ExactGP; …) … end`):

```julia
# Per-model fit hooks: training (inputs, targets) and rebuild+condition from a trial kernel.
_fit_xy(g::ExactGP) = (g.x, g.δ .+ AbstractGPs.mean(g.prior, g.x))
_recondition(g::ExactGP, kernel, X, y) = update(ExactGP(kernel; noise = g.noise, mean = g.prior.mean), X, y)

function fit(g::AbstractGPModel; restarts::Int = 1, ad = nothing, ℓ_prior = :auto)
    _outputdim(g) == 1 ||
        throw(ArgumentError("fit currently supports single-output models; got _outputdim=$(_outputdim(g))."))
    # Auto-wrap so a tunable σ_f² leaf is always present (a bare `with_lengthscale` has none).
    # Don't wrap composite (sum/product) kernels — their components already carry σ_f² leaves.
    k0 =
        g.prior.kernel isa KernelFunctions.ScaledKernel ||
        g.prior.kernel isa KernelFunctions.KernelSum ||
        g.prior.kernel isa KernelFunctions.KernelProduct ? g.prior.kernel : 1.0 * g.prior.kernel
    θ0, re = destructure(k0)
    (!isempty(θ0) && all(>(0), θ0)) || throw(
        ArgumentError(
            "fit optimizes positive scale hyperparameters (lengthscales, output scales) in log-space, " *
                "but the kernel destructured to $(θ0) — empty or with a non-positive leaf. Supported: " *
                "SqExponential/Matern/RationalQuadratic kernels (incl. ARD) and their sums/products. " *
                "Got $(typeof(g.prior.kernel)).",
        ),
    )
    if (k0 isa KernelFunctions.KernelSum || k0 isa KernelFunctions.KernelProduct) && !_has_scale(k0)
        @warn "fit: this composite kernel has no scale factor, so the signal variance σ_f² is fixed at 1 and not tuned. Add a scale factor (e.g. `1.0 * k`) to a component to calibrate it." maxlog = 1
    end
    ad === nothing && (ad = _default_ad(k0))
    X, y = _fit_xy(g)
    # MAP lengthscale prior: scalar-lengthscale kernels only (`_lengthscale` throws otherwise).
    scalar_ℓ = try
        (_lengthscale(k0); true)
    catch
        false
    end
    if ℓ_prior === :auto
        pri = scalar_ℓ ? (log(_lengthscale(k0)), 0.75) : nothing
    elseif ℓ_prior === nothing
        pri = nothing
    else
        scalar_ℓ || throw(ArgumentError(
            "ℓ_prior=(μ,σ) requires a scalar-lengthscale kernel; this one is ARD/composite. Use ℓ_prior=nothing.",
        ))
        pri = ℓ_prior
    end
    logθ0 = log.(θ0)
    np = length(logθ0)
    function loss(logθ, _)
        k = re(exp.(logθ))
        base = nlml(_recondition(g, k, X, y))
        pen = pri === nothing ? zero(eltype(logθ)) : 0.5 * ((log(_lengthscale(k)) - pri[1]) / pri[2])^2
        return base + pen
    end
    obj(logθ) = loss(logθ, nothing)
    best = g; best_obj = obj(logθ0)
    for r in 1:restarts
        start = r == 1 ? logθ0 : logθ0 .+ 0.1 .* randn(np)
        prob = OptimizationProblem(OptimizationFunction(loss, ad), start; lb = fill(-6.0, np), ub = fill(6.0, np))
        sol = solve(prob, LBFGS())
        if obj(sol.u) < best_obj
            best = _recondition(g, re(exp.(sol.u)), X, y)
            best_obj = obj(sol.u)
        end
    end
    return best
end
```

(This is the existing `ExactGP` `fit` body with exactly three substitutions: the signature `ExactGP → AbstractGPModel`, the `g.d == 1` guard → `_outputdim(g) == 1`, and the inlined data-extraction / `update(ExactGP(...))` calls → the `_fit_xy` / `_recondition` hooks. For an `ExactGP` it is behaviorally identical. Update the `fit` docstring's first line to read `fit(g::AbstractGPModel; …)` and add a sentence: "Works for any `AbstractGPModel` that defines `nlml`, `_fit_xy`, and `_recondition` — `ExactGP` (exact evidence) and `LaplaceGP` (Laplace evidence).")

- [ ] **Step 5: Run the recovery test, the Laplace tests, then the full suite**

Run: `julia --project=. test/test_laplace.jl` → Expected: PASS (recovery test + Task-1 tests).
Run: `julia --project=. test/test_fit.jl` → Expected: PASS (ExactGP fit unchanged).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green. The `ActiveLearner` `refit_every` path (`src/loop.jl:107`) now actually refits a `LaplaceGP` instead of no-op'ing; confirm `test_loop.jl` and `test_exemplar_A2.jl` stay green (A2 uses no refit, so it is unaffected; if any LaplaceGP loop test sets `refit_every>0`, it now exercises real fitting).

- [ ] **Step 6: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add src/fit.jl src/laplace.jl test/test_laplace.jl
git commit -m "feat(fit): generic fit over AbstractGPModel — LaplaceGP fits via Laplace evidence"
```

---

## Self-Review

**Spec coverage:** (this plan implements the "do LaplaceGP fit now" decision) — `nlml(::LaplaceGP)` Laplace evidence (Task 1) ✓; generic `fit` dispatching on `nlml` via `_fit_xy`/`_recondition` hooks (Task 2) ✓; no-op removed (Task 2) ✓; loop refit now real (Task 2 Step 5 verifies) ✓; ExactGP byte-for-byte preserved (Task 2 note + `test_fit.jl` green) ✓; AD-feasibility spike-verified (recorded in Global Constraints) ✓.

**Placeholder scan:** none — every step has complete code and exact commands.

**Type consistency:** `nlml(::LaplaceGP) -> Real` matches the generic `nlml` used in `loss`. `_fit_xy(g) -> (X, y)` and `_recondition(g, kernel, X, y) -> model` have identical signatures for both `ExactGP` and `LaplaceGP`. `_outputdim` returns `Int` (existing). The generic `fit` returns the same model type it was given (`best = g` or `_recondition(g, …)`).

## Follow-on

After this lands, the examples-rework plan (Plan B) can make `bald_classification.jl` **fit** the classifier's (isotropic) lengthscale — a "fit adapts to the data" story — instead of a hand-set value.
