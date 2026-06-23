# Kernel Parameterization (destructure-based `fit`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fit` optimize the hyperparameters of *any* user-built KernelFunctions kernel (ARD, composite sums/products, RationalQuadratic) by `destructure`-ing it into a flat positive-scale vector, instead of the hardcoded `[logℓ, logσ²]` single-base-kernel reconstruction.

**Architecture:** `fit` treats the GP's kernel as a structural template: `Optimisers.destructure(k)` → flat leaf vector + `rebuild` closure; optimize `log` of the vector (all supported leaves are positive scales) with bounded LBFGS; `rebuild` at the optimum and re-condition. A bare `with_lengthscale(...)` kernel is auto-wrapped as `1.0 * k` so a signal-variance leaf is always present and tuned. The MAP lengthscale prior stays a scalar-lengthscale feature (computed via the existing `_lengthscale` peeler on the rebuilt trial kernel — AD-safe, spike-verified). The two other peeler consumers that intrinsically need a *scalar* lengthscale — `grad_predict`'s analytic Matérn prior-gradient-variance and `LocalPenalization`'s radius — are hardened to raise a clear `ArgumentError` on ARD/composite kernels instead of a cryptic `only()` failure.

**Tech Stack:** Julia ≥1.10, KernelFunctions.jl, AbstractGPs.jl, Optimization.jl + OptimizationOptimJL (LBFGS), DifferentiationInterface.jl (ForwardDiff/Mooncake), **Optimisers.jl** (`destructure`).

Source of truth: the [kernel-parameterization design note](../specs/2026-06-22-al-foundation-hardening-design.md) and especially [`2026-06-22-kernel-parameterization-design.md`](../specs/2026-06-22-kernel-parameterization-design.md). This plan resolves that note's three open decisions: **(1) destructure dep → `Optimisers.destructure`** (already a transitive dep — Manifest line 657 — so promotion is free, and it is exactly the AD-Spike-2 mechanism); **(2) source of truth → re-`destructure` `g.prior.kernel` each `fit`** (no new struct field); **(3) AD backend → keep the metric-based `_default_ad`** (verified ForwardDiff-clean through `re`).

## Global Constraints

- **Use `Optimisers.destructure`** (promote `Optimisers` to a direct dep in `Project.toml`); do not hand-roll a flatten/rebuild helper.
- **`fit` auto-wraps** a non-`ScaledKernel` kernel as `1.0 * k` before destructuring, so the signal variance σ_f² is always a tunable leaf — fit must keep calibrating σ_f² (derivative/straddle acquisitions need it), matching today's behavior.
- **`fit` validates** the destructured vector is non-empty with **all positive** leaves; otherwise a clear `ArgumentError`. This replaces the `_kernelfamily` whitelist (which is deleted).
- **The `:auto` lengthscale prior is scalar-lengthscale-only.** For ARD/composite kernels `:auto` falls back to no prior (pure MLE); an explicit `ℓ_prior=(μ,σ)` on a non-scalar kernel raises a clear `ArgumentError`. The output scale is never penalized. The prior penalty is `0.5·((log(_lengthscale(rebuilt_trial_kernel)) − μ)/σ)²` — identical in value to today's for scalar kernels.
- **`_default_ad` is unchanged** — ForwardDiff for `SqExponentialKernel` base (incl. ARD), Mooncake otherwise; both verified clean through `Optimisers.destructure`'s `re`.
- **`fit` stays single-output (d=1)** — keep the existing `g.d == 1 || throw(...)` guard (added in Phase 1.5).
- **`_lengthscale` is hardened** to raise a clear `ArgumentError` for ARD/composite/non-isotropic kernels (it backs `grad_predict`'s Matérn constant and `LocalPenalization`). `_outputscale` is unchanged (σ² is scalar regardless of ARD).
- **All existing tests stay green** — the 4 `test/test_fit.jl` testsets, the derivative/acquisition tests, and the A1/A2/critpoints exemplars must pass unchanged. Format with **Runic** before each commit; run the **full suite** before each commit.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `Project.toml` | add `Optimisers` direct dep (+ compat) | 1 |
| `src/fit.jl` | destructure-based `fit`; delete `_kernelfamily`; scalar-ℓ MAP prior via peeler; rewrite docstring | 1 |
| `src/fit.jl` | harden `_lengthscale` dispatch (clear error on ARD/composite) | 2 |
| `test/test_fit.jl` | keep 4 existing testsets; add ARD-recovery + composite-fit | 1 |
| `test/test_ardkernels.jl` | NEW — `_lengthscale` error surface; `grad_predict`/`LocalPenalization` on ARD | 2 |
| `test/runtests.jl` | register the new test file | 2 |

`_lengthscale` / `_outputscale` / `_basekernel` are **kept** — they back the MAP prior (Task 1), `grad_predict` (`src/derivatives.jl`), and `LocalPenalization` (`src/acquisitions.jl`). Only `_kernelfamily` is deleted.

---

### Task 1: destructure-based `fit`

**Files:**
- Modify: `Project.toml` — add `Optimisers` to `[deps]`
- Modify: `src/fit.jl` — top imports (line 1); delete `_kernelfamily` (lines 13-16); rewrite `fit` (lines 42-101) incl. docstring
- Modify: `test/test_fit.jl` — append two testsets (keep the 4 existing ones)

**Interfaces:**
- Consumes: `ExactGP`, `nlml`, `update`, `_lengthscale`, `_outputscale`, `_basekernel`, `_default_ad` (all existing in `src/fit.jl`); `Optimisers.destructure`.
- Produces: `fit(g::ExactGP; restarts=1, ad=nothing, ℓ_prior=:auto) -> ExactGP` that optimizes any positive-scale kernel; `_kernelfamily` no longer exists.

- [ ] **Step 1: Add the dependency**

Run (this edits `Project.toml` `[deps]` + `[compat]` and `Manifest.toml`; `Optimisers` is already resolved at v0.4.x transitively, so this is fast):

```bash
julia --project=. -e 'using Pkg; Pkg.add("Optimisers")'
```

Expected: `+ Optimisers vX.Y.Z` added to Project.toml and Manifest.toml.

- [ ] **Step 2: Write the failing tests** — append to `test/test_fit.jl` (before the final `end`-of-file; do NOT modify the 4 existing testsets):

```julia
@testset "fit recovers ARD (per-dimension) lengthscales" begin
    # Anisotropic target: oscillates in x1 (short ℓ), nearly flat in x2 (long ℓ).
    # fit must drive the x1 inverse-lengthscale ABOVE the x2 one.
    Random.seed!(11)
    f(x) = sinpi(2x[1]) + 0.1 * x[2]
    X = [2 .* rand(2) .- 1 for _ in 1:60]
    y = f.(X) .+ 1.0e-3 .* randn(60)
    k0 = 1.0 * with_lengthscale(SqExponentialKernel(), [0.5, 0.5])    # ARD, isotropic start
    g0 = Magpie.update(ExactGP(k0; noise = 1.0e-3), X, y)
    g = fit(g0)
    invℓ = g.prior.kernel.kernel.transform.s     # ScaledKernel → TransformedKernel → ARDTransform.s
    @test invℓ[1] > invℓ[2]                       # x1 lengthscale shorter (larger inverse) than x2
    @test nlml(g) ≤ nlml(g0) + 1.0e-6
end

@testset "fit handles a composite (sum) kernel" begin
    Random.seed!(12)
    X = [randn(2) for _ in 1:40]; y = [sum(abs2, xi) for xi in X]
    k0 = 1.0 * with_lengthscale(SqExponentialKernel(), 0.8) +
        1.0 * with_lengthscale(Matern32Kernel(), 0.8)
    g0 = Magpie.update(ExactGP(k0; noise = 1.0e-3), X, y)
    g = fit(g0)
    @test nlml(g) ≤ nlml(g0) + 1.0e-6                    # composite fit improved the objective
    @test g.prior.kernel isa KernelFunctions.KernelSum   # structure preserved through destructure/rebuild
end
```

- [ ] **Step 3: Run them to verify they fail**

Run: `julia --project=. test/test_fit.jl`
Expected: FAIL — the current `fit` calls `_kernelfamily(_basekernel(k))`, which throws `ArgumentError` for the ARD/composite kernels (and `_lengthscale` would `only()`-fail on the ARD vector), so the new testsets error out.

- [ ] **Step 4: Add the `Optimisers.destructure` import** — at the top of `src/fit.jl`, change line 1 from:

```julia
using Optimization, OptimizationOptimJL, DifferentiationInterface
```

to:

```julia
using Optimization, OptimizationOptimJL, DifferentiationInterface
using Optimisers: destructure
```

- [ ] **Step 5: Delete `_kernelfamily`** — remove these lines from `src/fit.jl` (currently lines 13-16):

```julia
_kernelfamily(::SqExponentialKernel) = SqExponentialKernel()
_kernelfamily(::Matern32Kernel) = Matern32Kernel()
_kernelfamily(::Matern52Kernel) = Matern52Kernel()
_kernelfamily(k) = throw(ArgumentError("fit supports SqExponential/Matern32/Matern52 base kernels; got $(typeof(k)). (grad_predict's derivative path covers the same set.)"))
```

Keep `_lengthscale`, `_outputscale`, `_basekernel`, and `_default_ad` (lines 5-11, 21) exactly as they are — they back the MAP prior here and `grad_predict`/`LocalPenalization` elsewhere.

- [ ] **Step 6: Rewrite `fit`** — replace the entire `fit` function and its docstring (currently lines 42-101) with:

````julia
"""
    fit(g::ExactGP; restarts=1, ad=nothing, ℓ_prior=:auto) -> ExactGP

Optimize the kernel's hyperparameters by minimizing [`nlml`](@ref) (plus a lengthscale prior;
see below) with LBFGS, returning a GP re-conditioned at the best hyperparameters.

`fit` treats the GP's kernel as a structural template: it `Optimisers.destructure`s it into a
flat vector of positive scale parameters (inverse-lengthscales and output scales), optimizes the
**log** of that vector (bounded to `[-6, 6]`, so every scale stays positive), and rebuilds the
kernel. This supports **any** KernelFunctions kernel whose hyperparameters are positive scales:
`SqExponential`/`Matern`/`RationalQuadratic` bases, **ARD** (a lengthscale per input dimension),
and **sums/products** of these. A bare `with_lengthscale(base, ℓ)` kernel (no signal-variance
factor) is auto-wrapped as `1.0 * k` so σ_f² is always a tunable leaf — fitting σ_f² calibrates
the function scale, which the derivative/straddle acquisitions need.

`ad` selects the DifferentiationInterface backend; `nothing` (default) picks automatically:
`AutoForwardDiff()` for `SqExponentialKernel` bases (fast, smooth at `r=0`) and `AutoMooncake()`
otherwise (Matérn bases NaN under ForwardDiff at coincident points via `sqrt(0)`; composites
default to Mooncake too). With `restarts > 1`, extra runs start from the initial point jittered
in log-space and the result with the lowest **penalized** objective wins.

## Lengthscale prior (MAP, default on — scalar-lengthscale kernels only)

For a single-scalar-lengthscale kernel, `fit` is MAP: it adds a weakly-informative Gaussian prior
on `logℓ`, `0.5·((logℓ − μ)/σ)²`. **`ℓ_prior=:auto`** (default) centres it on the **initial
lengthscale** of `g`'s kernel with width `σ=0.75` (log units) — *refine the lengthscale you
specified, don't run away from it.* This matters when data is scarce: pure MLE drives `ℓ` up (a
flat surface explains few points cheaply), over-smoothing away the wells/saddles one hunts. Pass
`ℓ_prior=(μ, σ)` to set centre/width explicitly, or `ℓ_prior=nothing` for pure MLE. σ_f² is never
penalized.

For **ARD or composite** kernels there is no single lengthscale, so `ℓ_prior=:auto` falls back to
**no prior** (pure MLE); passing an explicit `ℓ_prior=(μ,σ)` then raises an `ArgumentError`.

!!! note
    `fit` requires the kernel's hyperparameters to be **positive scales** (lengthscales, output
    scales). Kernels with non-positive or non-scale leaves (e.g. `LinearKernel`, whose offset
    destructures to `0.0`) raise an `ArgumentError`.
"""
function fit(g::ExactGP; restarts::Int = 1, ad = nothing, ℓ_prior = :auto)
    g.d == 1 ||
        throw(ArgumentError("fit currently supports single-output GPs (d=1); got d=$(g.d)."))
    # Auto-wrap so a tunable σ_f² leaf is always present (a bare `with_lengthscale` has none).
    k0 = g.prior.kernel isa KernelFunctions.ScaledKernel ? g.prior.kernel : 1.0 * g.prior.kernel
    θ0, re = destructure(k0)
    (!isempty(θ0) && all(>(0), θ0)) || throw(ArgumentError(
        "fit optimizes positive scale hyperparameters (lengthscales, output scales) in log-space, " *
            "but the kernel destructured to $(θ0) — empty or with a non-positive leaf. Supported: " *
            "SqExponential/Matern/RationalQuadratic kernels (incl. ARD) and their sums/products. " *
            "Got $(typeof(g.prior.kernel)).",
    ))
    ad === nothing && (ad = _default_ad(k0))
    X = g.x; y = g.δ .+ AbstractGPs.mean(g.prior, g.x)
    noise = g.noise; meanfn = g.prior.mean
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
    # logθ → kernel via the rebuild closure; penalty reads logℓ off the rebuilt kernel (AD-safe).
    function loss(logθ, _)
        k = re(exp.(logθ))
        base = nlml(update(ExactGP(k; noise = noise, mean = meanfn), X, y))
        pen = pri === nothing ? zero(eltype(logθ)) : 0.5 * ((log(_lengthscale(k)) - pri[1]) / pri[2])^2
        return base + pen
    end
    obj(logθ) = loss(logθ, nothing)
    best = g; best_obj = obj(logθ0)                                  # penalty(logθ0)=0 for :auto
    for r in 1:restarts
        start = r == 1 ? logθ0 : logθ0 .+ 0.1 .* randn(np)           # first run exact, rest jittered
        prob = OptimizationProblem(OptimizationFunction(loss, ad), start; lb = fill(-6.0, np), ub = fill(6.0, np))
        sol = solve(prob, LBFGS())
        if obj(sol.u) < best_obj
            best = update(ExactGP(re(exp.(sol.u)); noise = noise, mean = meanfn), X, y)
            best_obj = obj(sol.u)
        end
    end
    return best
end
````

- [ ] **Step 7: Run the new tests, then the full suite**

Run: `julia --project=. test/test_fit.jl`
Expected: PASS — all 6 testsets (4 existing + ARD + composite). The 4 existing pass because: scalar kernels still destructure to `[inv(ℓ), σ²]` and rebuild to the same `ScaledKernel(TransformedKernel(...))` form `_lengthscale`/`_outputscale` read; the `:auto` prior value is identical to before; and `LinearKernel` still destructures to a non-positive leaf so `@test_throws` still fires.

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: green (≈136+2 tests). The A1/A2/critpoints exemplars must be unchanged — they fit scalar-lengthscale kernels, so the prior and σ_f² tuning behave exactly as before.

- [ ] **Step 8: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add Project.toml Manifest.toml src/fit.jl test/test_fit.jl
git commit -m "feat(fit): destructure-based hyperparameter fitting (ARD, composite kernels)"
```

---

### Task 2: clear errors on scalar-lengthscale-required paths

`grad_predict`'s analytic Matérn prior-gradient-variance (`_prior_grad_var_const`) and
`LocalPenalization`'s radius both call `_lengthscale`, which is defined for a *single scalar*
lengthscale. On an ARD/composite kernel the current `1 / only(k.transform.s)` throws a cryptic
"Collection has multiple elements" or "type … has no field transform". Harden `_lengthscale` to a
typed dispatch that raises a clear `ArgumentError`. (ARD `SqExponential` `grad_predict` still
works — its prior-gradient-variance is computed by AD, not the peeler.)

**Files:**
- Modify: `src/fit.jl` — `_lengthscale` definitions (currently lines 5-6)
- Create: `test/test_ardkernels.jl`
- Modify: `test/runtests.jl` — register the new file

**Interfaces:**
- Consumes: `grad_predict`, `LocalPenalization`, `Straddle`, `_lengthscale` (existing).
- Produces: `_lengthscale` raises a clear `ArgumentError` on ARD/composite/non-isotropic kernels; scalar usage unchanged.

- [ ] **Step 1: Write the failing test** — create `test/test_ardkernels.jl`:

```julia
using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, update, grad_predict, _lengthscale, LocalPenalization, Straddle

@testset "_lengthscale: clear error on ARD/composite, scalar still works" begin
    @test_throws ArgumentError _lengthscale(with_lengthscale(SqExponentialKernel(), [0.5, 1.0]))
    @test_throws ArgumentError _lengthscale(with_lengthscale(SqExponentialKernel(), 0.5) + Matern32Kernel())
    @test _lengthscale(2.0 * with_lengthscale(SqExponentialKernel(), 0.7)) ≈ 0.7
    @test _lengthscale(with_lengthscale(Matern52Kernel(), 1.3)) ≈ 1.3
end

@testset "grad_predict: works for ARD SqExp, clear error for ARD Matern" begin
    Random.seed!(21)
    X = [randn(2) for _ in 1:12]; y = [sum(abs2, xi) for xi in X]
    g_sq = update(ExactGP(1.0 * with_lengthscale(SqExponentialKernel(), [0.5, 0.8]); noise = 1.0e-4), X, y)
    μ∇, Σ, H = grad_predict(g_sq, randn(2))                 # smooth kernel: AD path, no scalar-ℓ needed
    @test length(μ∇) == 2 && all(isfinite, μ∇) && all(isfinite, Σ)
    g_m = update(ExactGP(1.0 * with_lengthscale(Matern32Kernel(), [0.5, 0.8]); noise = 1.0e-4), X, y)
    @test_throws ArgumentError grad_predict(g_m, randn(2))  # ARD Matern → _prior_grad_var_const → _lengthscale
end

@testset "LocalPenalization: clear error on an ARD kernel" begin
    Random.seed!(22)
    X = [randn(2) for _ in 1:8]; y = [sum(xi) for xi in X]
    g = update(ExactGP(1.0 * with_lengthscale(SqExponentialKernel(), [0.5, 0.8]); noise = 1.0e-4), X, y)
    lp = LocalPenalization(Straddle(h = 0.0), [randn(2)])   # non-empty pts → reads ℓ
    @test_throws ArgumentError lp(g, randn(2))
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_ardkernels.jl`
Expected: FAIL — `_lengthscale` on the ARD kernel throws an `ArgumentError` from `only()` (so that `@test_throws` may pass by accident), but `_lengthscale` on the **composite sum** throws a non-`ArgumentError` (`type KernelSum has no field transform` → `ErrorException`/`FieldError`), failing `@test_throws ArgumentError`. The grad_predict/LocalPenalization assertions likewise depend on a *clear* error.

- [ ] **Step 3: Harden `_lengthscale`** — in `src/fit.jl`, replace lines 5-6:

```julia
_lengthscale(k) = 1 / only(k.transform.s)
_lengthscale(k::KernelFunctions.ScaledKernel) = _lengthscale(k.kernel)
```

with typed dispatch:

```julia
_lengthscale(k::KernelFunctions.ScaledKernel) = _lengthscale(k.kernel)
_lengthscale(k::KernelFunctions.TransformedKernel) = _ls_from_transform(k.transform)
_lengthscale(k) = throw(ArgumentError(
    "no scalar lengthscale for a kernel of type $(nameof(typeof(k))); grad_predict's analytic " *
        "prior-gradient-variance and LocalPenalization's radius require an isotropic " *
        "`with_lengthscale` kernel (optionally scaled).",
))
_ls_from_transform(t::KernelFunctions.ScaleTransform) = 1 / only(t.s)
_ls_from_transform(t) = throw(ArgumentError(
    "no scalar lengthscale for an ARD/$(nameof(typeof(t))) transform; grad_predict and " *
        "LocalPenalization are scalar-lengthscale only.",
))
```

(Comment above line 5 — "Peel ScaledKernel/TransformedKernel wrappers to read hyperparameters." — stays accurate; leave it.)

- [ ] **Step 4: Register the new test file** — in `test/runtests.jl`, add after the `test_multioutput.jl` include:

```julia
    include("test_ardkernels.jl")
```

- [ ] **Step 5: Run the new test, then the full suite**

Run: `julia --project=. test/test_ardkernels.jl` → Expected: PASS.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green. Scalar `_lengthscale`/`_outputscale` usage (`test_derivatives.jl`, `test_fit.jl`, `LocalPenalization` in `test_loop.jl`) is unchanged — `with_lengthscale`/`ScaledKernel` kernels dispatch to the working methods.

- [ ] **Step 6: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add src/fit.jl test/test_ardkernels.jl test/runtests.jl
git commit -m "feat(fit): clear errors when grad_predict/LocalPenalization meet non-scalar-lengthscale kernels"
```

---

## Self-Review

**Spec coverage** (kernel-parameterization design note → tasks):
- `fit` destructure-based, replacing hardcoded `mkkernel` → **Task 1** ✓
- log-transform constraint layer (`log`/`exp` of positive leaves) → **Task 1** ✓ (with all-positive validation)
- migrate `LocalPenalization`/`grad_predict` off the scalar-`_lengthscale` assumption → **Task 2** ✓ (they intrinsically *need* a scalar ℓ — analytic Matérn derivative / radius — so the honest migration is a clear error, not silent ARD support; ARD SqExp `grad_predict` already works via AD and is tested)
- metric-based `_default_ad` folded in → already present from Phase 1; verified ForwardDiff-clean through `re` (spike); **no change needed** ✓
- Open decision 1 (dep) → `Optimisers.destructure`, **Task 1 Step 1** ✓
- Open decision 2 (source of truth) → re-`destructure` each fit, no struct field → **Task 1** ✓
- Open decision 3 (AD backend) → keep `_default_ad` ✓

**Placeholder scan:** none — every step has complete code and exact commands.

**Type consistency:** `destructure(k0)` returns `(θ0::Vector{Float64}, re)`; `re(exp.(logθ))::Kernel`; `_lengthscale(::Kernel)::Float64` (or throws); `_default_ad(::Kernel)` → an `AutoForwardDiff`/`AutoMooncake`. `loss(logθ, _)` matches Optimization.jl's `(u, p)` signature. Bounds `fill(-6.0, np)`/`fill(6.0, np)` match `length(logθ0)`. The ARD test reads `g.prior.kernel.kernel.transform.s` — valid because `re` rebuilds `ScaledKernel(TransformedKernel(base, ARDTransform(s)), σ²)`.

## Risks (retired by spikes)

- **ForwardDiff through `re`** — verified finite, relerr 2.8e-8 (scalar SqExp) and finite (ARD SqExp).
- **MAP prior `log(_lengthscale(re(...)))` differentiability** — verified ForwardDiff & Mooncake clean, relerr 5e-11; value equals `logℓ`.
- **`destructure` round-trip fidelity** — design-note Spike 2: rebuild reproduces the kernel exactly, 6-param ARD+composite, no spurious params.

## Follow-on (separate plans)

- **Multi-output fitting** — lift Phase 1.5's `fit` `d>1` guard once a multi-output objective is defined; would also be where ARD-per-output is considered.
- **Bounded hyperparameters** (e.g. `GammaExponentialKernel`'s γ∈(0,2], `RationalQuadratic`'s α if not treated as a positive scale) — the log-transform assumes positive-unbounded; add a per-leaf transform layer when such a kernel is first needed.
- **ARD/composite `grad_predict`** — analytic Matérn prior-gradient-variance for ARD (new per-family math) if derivative-based acquisitions are ever wanted on anisotropic kernels.
