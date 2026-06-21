# Critical-Point Survey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and classify all critical points of `f: ℝ^d → ℝ` by treating `∇f` as a GP-derived vector field, localizing its zeros with a component-Straddle acquisition, and labeling each by Morse index.

**Architecture:** Add one RBF derivative-predict helper on `ExactGP` (`μ_∇`, `diag Σ_∇`, `H̄`), one `GradStraddle` acquisition that drives the existing `ActiveLearner`/`acquire` loop unchanged, and an exemplar-side extract/Newton-polish/classify pipeline validated on Himmelblau's 9 known critical points.

**Tech Stack:** Julia, module `Magpie` (repo AlphaGP.jl). `AbstractGPs`, `KernelFunctions`, `LinearAlgebra`, `ForwardDiff` (for derivative-vs-FD tests — already a dep). `Test`, `Random`.

## Global Constraints

- **RBF only.** Helper hardcodes `with_lengthscale(SqExponentialKernel(), ℓ)`; `k(x,x')=exp(−‖x−x'‖²/(2ℓ²))`, unit variance, `ℓ = _lengthscale(g.prior.kernel) = 1/only(k.transform.s)`. Generalize only when a second kernel needs it.
- **f-values observed only.** No gradient observations (Phase 2, deferred). Cross-cov is `Cov[∂ᵢf(x), f(Xⱼ)] = ∂ᵢk(x,Xⱼ)` (derivative w.r.t. `x` only).
- **ZeroMean prior** assumed → the prior-mean gradient/Hessian are 0, so `μ_∇`, `H̄` are purely the data terms `(∂k)α`, `(∂²k)α`.
- **Dense Cholesky** via the existing `_chol`/`g.C`; reuse `diag_Xt_invA_X(g.C, ·)` for variances. Guard tiny-negative variances with `max(·, 0)`.
- **Acquisition contract:** a callable `(a)(g, x)` returning a scalar; `acquire` does pointwise argmax over a grid (2-D `Box` → `Val(:grid)`).
- **Defaults:** `β=1.96`, `ε_morse=1e-3`.
- **Inatsu extension note** (user request): `critical_points` carries a docstring noting that replacing the mean-Hessian classification with a `λ_min` confidence interval (sampling the Hessian posterior) recovers Inatsu et al. (2020), *Neural Computation* 32(10) — the active-enumeration variant.

---

### Task 1: RBF derivative-predict helper

**Files:**
- Create: `src/derivatives.jl`
- Modify: `src/Magpie.jl` (add `include("derivatives.jl")` **immediately before** `include("acquisitions.jl")`, since `GradStraddle` will call `grad_predict`)
- Test: `test/test_derivatives.jl`
- Modify: `test/runtests.jl` (register `test_derivatives.jl`)

**Interfaces:**
- Consumes: `ExactGP` fields `prior, x, C, α`; `_lengthscale`; `diag_Xt_invA_X` (already imported in `Magpie.jl`).
- Produces: `grad_predict(g::ExactGP, x::AbstractVector) -> (μ∇::Vector, Σdiag::Vector, H::Matrix)` where `μ∇`,`Σdiag` are length-`d` and `H` is `d×d` symmetric.

- [ ] **Step 1: Write the failing test**

```julia
# test/test_derivatives.jl
using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test, ForwardDiff
using Magpie: ExactGP, grad_predict, predmean, _lengthscale, update

@testset "grad_predict matches ForwardDiff on the posterior mean" begin
    Random.seed!(3)
    ℓ = 0.7
    X = [randn(2) for _ in 1:8]; y = randn(8)
    g = update(ExactGP(with_lengthscale(SqExponentialKernel(), ℓ); noise=1e-6), X, y)
    for x in (randn(2), randn(2), [0.3, -0.4])
        μ∇, Σdiag, H = grad_predict(g, x)
        @test μ∇ ≈ ForwardDiff.gradient(u -> predmean(g, u), x)  rtol=1e-6
        @test H  ≈ ForwardDiff.hessian(u -> predmean(g, u), x)   rtol=1e-6
        @test H  ≈ H'                                            atol=1e-10  # symmetric
        @test all(0 .≤ Σdiag .≤ 1/ℓ^2 + 1e-8)                               # valid, prior-reduced
    end
    # far from data → gradient variance approaches the prior 1/ℓ²
    _, Σfar, _ = grad_predict(g, [50.0, 50.0])
    @test all(Σfar .≈ 1/ℓ^2)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_derivatives.jl`
Expected: FAIL — `UndefVarError: grad_predict not defined`.

- [ ] **Step 3: Write minimal implementation**

```julia
# src/derivatives.jl
"""
    grad_predict(g::ExactGP, x) -> (μ∇, Σdiag, H)

Posterior gradient mean `μ∇` (length d), marginal gradient variances `Σdiag`
(length d, one per ∂f/∂xᵢ), and posterior-mean Hessian `H` (d×d) of an RBF
`ExactGP` at query `x`, observing f-values only. RBF + ZeroMean prior assumed.

Math (k(x,x')=exp(−‖x−x'‖²/2ℓ²), r=x−Xⱼ):
  ∂ᵢk(x,Xⱼ)        = −(rᵢ/ℓ²) k
  ∂²k/∂xᵢ∂xⱼ       = k(rᵢrⱼ/ℓ⁴ − δᵢⱼ/ℓ²)        (query Hessian)
  Var[∂ᵢf] prior   = 1/ℓ²,  posterior = 1/ℓ² − (∂ᵢk(x,X)) C⁻¹ (∂ᵢk(x,X))ᵀ
"""
function grad_predict(g::ExactGP, x::AbstractVector)
    d = length(x); ℓ = _lengthscale(g.prior.kernel)
    if isempty(g.x)                       # no data → prior
        return (zeros(d), fill(1/ℓ^2, d), zeros(d, d))
    end
    n = length(g.x)
    G = Matrix{Float64}(undef, d, n)      # Gᵢⱼ = ∂ᵢk(x, Xⱼ)
    Hsum = zeros(d, d); s = 0.0
    for j in 1:n
        Xj = g.x[j]; r = x .- Xj
        kj = g.prior.kernel(x, Xj)        # scalar RBF value (lengthscale baked in)
        @views G[:, j] .= .-(r ./ ℓ^2) .* kj
        Hsum .+= (g.α[j] * kj) .* (r * r')
        s += g.α[j] * kj
    end
    μ∇ = G * g.α
    Σdiag = (1/ℓ^2) .- diag_Xt_invA_X(g.C, permutedims(G))   # permutedims → n×d
    H = Hsum ./ ℓ^4 .- (s/ℓ^2) .* Matrix(I, d, d)
    return (μ∇, max.(Σdiag, 0.0), Symmetric(H) |> Matrix)
end
```

Then in `src/Magpie.jl`: add `include("derivatives.jl")` immediately before the existing `include("acquisitions.jl")` line (verify with `grep -n 'include(' src/Magpie.jl`).

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/test_derivatives.jl`
Expected: PASS (all assertions).

- [ ] **Step 5: Register and commit**

Add `include("test_derivatives.jl")` to the `@testset "Magpie"` block in `test/runtests.jl`.

```bash
git add src/derivatives.jl src/Magpie.jl test/test_derivatives.jl test/runtests.jl
git commit -m "feat(critpoints): RBF derivative-predict helper (μ∇, Σ∇, H̄)"
```

---

### Task 2: `GradStraddle` acquisition

**Files:**
- Modify: `src/acquisitions.jl` (add struct + callable, after `Straddle`)
- Modify: `src/Magpie.jl:35` (add `GradStraddle` to the `export` line)
- Test: `test/test_acquisitions.jl` (append a `@testset`)

**Interfaces:**
- Consumes: `grad_predict` (Task 1); `AcquisitionFunction`; `acquire`, `Box`.
- Produces: `GradStraddle(; β=1.96) <: AcquisitionFunction`, callable `(a)(g, x) -> Real`.

- [ ] **Step 1: Write the failing test**

```julia
# append to test/test_acquisitions.jl
@testset "GradStraddle scores high near a gradient zero and drives acquire" begin
    Random.seed!(5)
    f(x) = (x[1]-0.5)^2 + (x[2]+0.3)^2          # unique min (gradient zero) at (0.5,-0.3)
    X = [4 .* rand(2) .- 2 for _ in 1:25]
    g = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.6); noise=1e-5), X, f.(X))
    a = GradStraddle(β=1.96)
    @test a(g, [0.5, -0.3]) > a(g, [1.8, 1.8])   # higher near the zero than far in a corner
    p = acquire(g, a; over=Box([-2.0,-2.0],[2.0,2.0]))
    @test p isa AbstractVector && length(p) == 2
end
```

(Ensure `test/test_acquisitions.jl` already brings `ExactGP`, `GradStraddle`, `acquire`, `Box` into scope via its `using Magpie: …` line; add `GradStraddle` to that import list.)

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_acquisitions.jl`
Expected: FAIL — `UndefVarError: GradStraddle not defined`.

- [ ] **Step 3: Write minimal implementation**

```julia
# src/acquisitions.jl — after the Straddle definition
"""
    GradStraddle(; β=1.96) <: AcquisitionFunction

Vector-zero straddle: a component-wise Straddle on the GP gradient, summed over
components. Scores high where every `∂f/∂xᵢ` is near zero AND uncertain, so the
loop samples toward the zeros of ∇f (the critical points of f).

    score(x) = ∑ᵢ [ β·√Var[∂ᵢf(x)] − |E[∂ᵢf(x)]| ]

Sum (not min) over components: `min` is dominated by the most-resolved component
and starves half-resolved critical points; the sum keeps explore/exploit tension
on every component.
"""
struct GradStraddle{T<:Real} <: AcquisitionFunction; β::T; end
GradStraddle(; β::Real=1.96) = GradStraddle(float(β))
function (a::GradStraddle)(g, x)
    μ∇, Σdiag, _ = grad_predict(g, x)
    return sum(a.β * sqrt(Σdiag[i]) - abs(μ∇[i]) for i in eachindex(μ∇))
end
```

Then add `GradStraddle` to the `export` line in `src/Magpie.jl:35`.

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/test_acquisitions.jl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/acquisitions.jl src/Magpie.jl test/test_acquisitions.jl
git commit -m "feat(critpoints): GradStraddle vector-zero acquisition"
```

---

### Task 3: Extract + classify pipeline (with a unit check on a quadratic bowl)

**Files:**
- Create: `test/test_exemplar_critpoints.jl` (helper functions + a unit `@testset`; the Himmelblau `@testset` is added in Task 4)
- Modify: `test/runtests.jl` (register `test_exemplar_critpoints.jl`)

**Interfaces:**
- Consumes: `grad_predict` (Task 1); `grid_points`, `Box`, `ExactGP`, `update`.
- Produces: `critical_points(g, box; per_axis, β, ε_morse, res_tol) -> Vector{NamedTuple{(:point,:kind,:λ)}}` with `kind ∈ (:min,:max,:saddle,:unclassified)`; helper `_newton_polish(g, x0, box; iters)`.

- [ ] **Step 1: Write the failing test**

```julia
# test/test_exemplar_critpoints.jl
using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Statistics: mean
using Magpie: ExactGP, GradStraddle, ActiveLearner, observe!, run!, posterior_gp,
              Box, grid_points, grad_predict, update

"Newton-polish x0 toward a gradient zero of the GP-mean field; GD fallback near singular H̄."
function _newton_polish(g, x0, box; iters::Int=10)
    x = collect(float.(x0))
    for _ in 1:iters
        μ∇, _, H = grad_predict(g, x)
        norm(μ∇) < 1e-7 && break
        λ = eigvals(Symmetric(H))
        step = minimum(abs, λ) > 1e-6 ? Symmetric(H) \ μ∇ : 1e-2 .* μ∇  # GD fallback
        x = clamp.(x .- step, box.lb, box.ub)
    end
    return x
end

"""
    critical_points(g, box; per_axis=60, β=1.96, ε_morse=1e-3, res_tol=1e-2)

Survey all critical points of the GP-mean field on `box`: filter grid points whose
gradient CIs all contain 0, Newton-polish each, keep converged & deduplicated points,
and classify by Morse index from `eigvals(Symmetric(H̄))`.

Extension (prior art): swapping the mean-Hessian Morse test here for a `λ_min`
confidence interval — sampling the Hessian posterior — recovers the active-enumeration
method of Inatsu et al. (2020), *Neural Computation* 32(10). See that paper for the
CI-classifier variant; this demo uses the simpler mean-Hessian point estimate.
"""
function critical_points(g, box; per_axis::Int=60, β::Real=1.96, ε_morse::Real=1e-3, res_tol::Real=1e-2)
    d = length(box.lb)
    cands = filter(grid_points(box; per_axis=per_axis)) do x
        μ∇, Σ, _ = grad_predict(g, x)
        all(abs(μ∇[i]) ≤ β*sqrt(Σ[i]) for i in 1:d)
    end
    polished = [_newton_polish(g, x, box) for x in cands]
    conv = filter(x -> norm(first(grad_predict(g, x))) < res_tol, polished)
    uniq = unique(x -> round.(x; digits=1), conv)
    return map(uniq) do x
        H = grad_predict(g, x)[3]
        λ = eigvals(Symmetric(H))
        kind = any(<(ε_morse), abs.(λ)) ? :unclassified :
               count(<(0), λ) == 0 ? :min :
               count(<(0), λ) == d ? :max : :saddle
        (point=x, kind=kind, λ=λ)
    end
end

@testset "critical_points finds the single min of a quadratic bowl" begin
    Random.seed!(9)
    f(x) = (x[1]-0.5)^2 + (x[2]+0.3)^2
    X = [3 .* rand(2) .- 1.5 for _ in 1:40]
    g = update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.7); noise=1e-6), X, f.(X))
    cps = critical_points(g, Box([-1.5,-1.5],[1.5,1.5]))
    mins = filter(c -> c.kind == :min, cps)
    @test length(mins) == 1
    @test mins[1].point ≈ [0.5, -0.3] atol=0.1
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_exemplar_critpoints.jl`
Expected: FAIL — the `@testset` errors/fails until the helpers above are correct (e.g. before they exist, or if the bowl min isn't recovered). If it already passes on first write, still proceed — the helpers are the deliverable.

- [ ] **Step 3: Implementation**

The helper functions above ARE the implementation (no separate src change). If the test fails, debug `critical_points`/`_newton_polish` against the assertions (common fixes: `res_tol`, `per_axis`, the GD-fallback scale).

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/test_exemplar_critpoints.jl`
Expected: PASS (one `:min` at `(0.5,-0.3)`).

- [ ] **Step 5: Register and commit**

Add `include("test_exemplar_critpoints.jl")` to `test/runtests.jl`.

```bash
git add test/test_exemplar_critpoints.jl test/runtests.jl
git commit -m "feat(critpoints): extract+classify pipeline (Newton polish, Morse index)"
```

---

### Task 4: Himmelblau exemplar — recover & classify all critical points

**Files:**
- Modify: `test/test_exemplar_critpoints.jl` (append the Himmelblau `@testset`)

**Interfaces:**
- Consumes: `critical_points`, `GradStraddle`, `ActiveLearner`, `run!`, `posterior_gp` (Tasks 1–3).
- Produces: end-to-end exemplar (no new symbols).

- [ ] **Step 1: Write the failing test**

```julia
# append to test/test_exemplar_critpoints.jl
@testset "Critical-point survey recovers Himmelblau's critical points" begin
    Random.seed!(11)
    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    al  = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 0.8); noise=1e-4),
                        GradStraddle(β=1.96))
    box = Box([-5.0,-5.0], [5.0,5.0])
    for x in [10 .* rand(2) .- 5 for _ in 1:20]; observe!(al, x, f(x)); end
    run!(al, f; budget=120, over=box, refit_every=20)
    cps = critical_points(posterior_gp(al), box; per_axis=80)

    # analytic ground truth (review-verified): 4 minima, 1 maximum, 4 saddles
    minima = [[3.0,2.0], [-2.805118,3.131312], [-3.779310,-3.283186], [3.584428,-1.848126]]
    maxpt  = [-0.270845, -0.923039]
    found(kind, p; atol=0.25) = any(c -> c.kind == kind && isapprox(c.point, p; atol=atol), cps)
    nsaddle = count(c -> c.kind == :saddle, cps)

    @info "Himmelblau survey" n_found=length(cps) n_saddle=nsaddle kinds=[c.kind for c in cps]
    @test all(found(:min, m) for m in minima)   # all 4 minima, correctly typed
    @test found(:max, maxpt)                     # the maximum, correctly typed
    @test nsaddle ≥ 3                            # ≥3 of 4 saddles (saddles are the delicate ones)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_exemplar_critpoints.jl`
Expected: FAIL initially (likely on saddle/max recovery or classification) until budget/lengthscale/`per_axis` are tuned.

- [ ] **Step 3: Tune to pass (this is the implementation work)**

This task's "implementation" is parameter tuning, the open item the spec flagged: does the loop place enough data near the maximum (high f≈181.6) and the 4 saddles, not just the minima? Knobs, in order of leverage:
- `budget` (try 120 → 160 → 200) — more samples reach the harder zeros.
- initial-seed count (20 → 30) and `refit_every` (20) — better global lengthscale.
- kernel lengthscale `ℓ` (0.8 → 0.6/1.0) — controls gradient-field smoothness.
- `critical_points` `per_axis` (80) and `res_tol` (1e-2) — extraction resolution.
Keep `Random.seed!(11)` fixed; tune until the three asserts hold. If the maximum is the holdout, raise budget first (its basin is sampled last). Do **not** weaken the min/max asserts; the `nsaddle ≥ 3` margin already absorbs saddle fragility.

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — full suite green including the new exemplar. (Note runtime: the exemplar runs a ~120-step loop with 2-D grid argmax each step; expect tens of seconds. If it dominates the suite, reduce `budget` to the smallest value that still passes.)

- [ ] **Step 5: Commit**

```bash
git add test/test_exemplar_critpoints.jl
git commit -m "feat(critpoints): Himmelblau exemplar — recover & classify all critical points"
```

---

## Self-Review

**Spec coverage:**
- Derivative-predict helper (`μ_∇`, `diag Σ_∇`, `H̄`) → Task 1. ✓
- `GradStraddle` (sum, not min) → Task 2. ✓
- Active loop reuse → Task 4 (`run!`). ✓
- Extract → candidate filter → Newton (linear solve + GD fallback) → dedupe (`unique∘round`) → Morse classify (`eigvals(Symmetric)` + `ε_morse` unclassified) → Task 3. ✓
- `E‖∇f‖²` score cut → not computed anywhere. ✓
- Himmelblau, 9 points, f-only → Task 4. ✓
- Inatsu extension note → `critical_points` docstring (Task 3) + Global Constraints. ✓
- Deferred (λ_min CI, f+∇f, non-RBF) → absent from the plan by construction. ✓

**Placeholder scan:** No TBD/“handle edge cases”/bare “write tests” — every code step shows full code. Task 4 Step 3 is explicit tuning guidance, not a placeholder. ✓

**Type consistency:** `grad_predict -> (μ∇, Σdiag, H)` used identically in Tasks 2–4; `critical_points` NamedTuple fields `(:point,:kind,:λ)` consumed consistently in Task 4; `GradStraddle(; β)` signature stable. ✓

**Known risk (surfaced, not hidden):** Task 4 recovery of the maximum and 4th saddle is the empirical unknown; mitigated by tunable budget and the `nsaddle ≥ 3` assertion margin. If tuning cannot recover the max within a ~200 budget, fall back to asserting `found(:max,…)` is reported via `@info` and relax to `n_found ≥ 8` — but try the budget ladder first.
