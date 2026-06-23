# Active-Learning Foundation Hardening — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the active-learning foundation correct and reproducible — input validation at every boundary, scale-invariant classification, gated convergence, kernel-generic fitting, threaded RNG, and type-stable storage — without reshaping the public API.

**Architecture:** In-place hardening of the existing Capability A surface (`spine.jl`, `fit.jl`, `acquisitions.jl`, `maximize.jl`, `loop.jl`, `saddle.jl`). Each task is an isolated, independently-testable correctness or table-stakes fix that keeps the full suite green. No module moves, no renames, no new dependencies. The multi-output spine (Phase 1.5) and the API/ergonomics reshaping (Phase 2) are **separate plans**.

**Tech Stack:** Julia ≥1.10, AbstractGPs.jl, KernelFunctions.jl, Optimization.jl/LBFGS, Mooncake (AD), DifferentiationInterface.jl, Sobol.jl.

Source of truth: [`docs/superpowers/specs/2026-06-22-al-foundation-hardening-design.md`](../specs/2026-06-22-al-foundation-hardening-design.md).

## Global Constraints

- **AD is Mooncake-first.** All matrix factorizations go through the `_chol` chokepoint in `src/spine.jl`: `_chol(K) = cholesky(Symmetric(K); check=false)`. Do not introduce a factorization outside it.
- **Julia ≥ 1.10.** `allequal`/`big` are available.
- **No new dependencies.** The AD test hand-rolls central differences (matching the existing `test/test_ad.jl` style) rather than adding FiniteDifferences.
- **Format with Runic before every commit:** `git ls-files -z -- '*.jl' | xargs -0 runic --inplace`.
- **The full suite must stay green** (baseline: 85 tests). Run `julia --project=. -e 'using Pkg; Pkg.test()'` before each commit; a single file runs standalone, e.g. `julia --project=. test/test_validation.jl`.
- **Commit conventions:** end commit messages with the repo's `Co-Authored-By:` / `Claude-Session:` trailer (omitted from the snippets below for brevity — copy from `git log`).

## File Structure

| File | Responsibility | Touched by |
|---|---|---|
| `src/spine.jl` | `ExactGP` + `_validate_obs` (new internal validator) | Task 1 |
| `src/laplace.jl` | `LaplaceGP` update — label coercion + validation | Task 1 |
| `src/maximize.jl` | `Box` inner constructor, `grid_points` high-D guard | Task 1 |
| `src/saddle.jl` | scale-invariant `classify`; walker `converged`/`residual`; RNG in `_ts_seed`/`transition_state` | Tasks 2, 3, 5 |
| `src/fit.jl` | kernel-generic `fit` | Task 4 |
| `src/acquisitions.jl` | `resample(a, rng)` primary methods | Task 5 |
| `src/loop.jl` | typed `ActiveLearner{TX,TY}` + `rng` field; thread rng | Task 6 |
| `test/test_validation.jl` | NEW — validation tests | Task 1 |
| `test/test_saddle.jl` | classify scale-invariance + convergence-flag + rng tests | Tasks 2, 3, 5 |
| `test/test_ad.jl` | Mooncake-vs-FD on the two-parameter penalized loss | Task 4 |
| `test/test_loop.jl` | typed storage + reproducibility tests | Task 6 |
| `test/test_exemplar_critpoints.jl` | rewrite the superseded exemplar | Task 7 |
| `test/runtests.jl` | add `include("test_validation.jl")` | Task 1 |

---

### Task 1: Input validation at every public boundary

**Files:**
- Modify: `src/spine.jl` (add `_validate_obs` + helpers near `update`, ~line 92; call it in `update`)
- Modify: `src/laplace.jl` (`update`, lines 74-79: accept `<:Real` labels, coerce, validate)
- Modify: `src/maximize.jl` (`Box`, line 18: inner constructor; `grid_points`, lines 55-58: high-D guard)
- Modify: `src/loop.jl` (`observe!`, lines 43-49: validate after batching)
- Create: `test/test_validation.jl`
- Modify: `test/runtests.jl` (add include)

**Interfaces:**
- Produces: `Magpie._validate_obs(X, y) -> nothing` (throws `ArgumentError`); `Magpie._to_labels(y) -> Vector{Bool}`; `Box` now validates; `grid_points` guards size.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test** — create `test/test_validation.jl`:

```julia
using Magpie, KernelFunctions, Test
using Magpie: ExactGP, LaplaceGP, update, Box, grid_points

@testset "observation validation" begin
    g = ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1e-4)
    @test_throws ArgumentError update(g, [[0.0], [1.0]], [0.0])          # count mismatch
    @test_throws ArgumentError update(g, Vector{Float64}[], Float64[])    # empty
    @test_throws ArgumentError update(g, [[0.0], [NaN]], [0.0, 1.0])      # non-finite input
    @test_throws ArgumentError update(g, [[0.0], [1.0]], [0.0, Inf])      # non-finite value
    @test_throws ArgumentError update(g, [[0.0], [0.0, 1.0]], [0.0, 1.0]) # inconsistent dim
end

@testset "Box validation" begin
    @test_throws ArgumentError Box([0.0, 0.0], [1.0])      # length mismatch
    @test_throws ArgumentError Box([1.0, 0.0], [0.0, 1.0]) # lb > ub at dim 1
end

@testset "grid_points high-D guard" begin
    @test_throws ArgumentError grid_points(Box(fill(-1.0, 5), fill(1.0, 5)); per_axis = 50)
end

@testset "LaplaceGP label coercion" begin
    g = LaplaceGP(with_lengthscale(SqExponentialKernel(), 0.5))
    X = [[0.0], [1.0], [2.0]]
    @test update(g, X, [0, 1, 1]) isa LaplaceGP        # {0,1} integers
    @test update(g, X, [-1, 1, -1]) isa LaplaceGP      # {-1,+1}
    @test update(g, X, [true, false, true]) isa LaplaceGP
    @test_throws ArgumentError update(g, X, [0, 2, 1])  # invalid label set
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_validation.jl`
Expected: FAIL/ERROR — `Box`/`update`/`grid_points` do not yet throw `ArgumentError` (e.g. count mismatch currently errors deep in `vcat`/broadcast, label `[0,2,1]` is a `MethodError`, not `ArgumentError`).

- [ ] **Step 3: Add `_validate_obs` and helpers to `src/spine.jl`** (insert just above `update`, ~line 92):

```julia
_allfinite(v::Number) = isfinite(v)
_allfinite(v) = all(isfinite, v)
_inputdim(v::Number) = 1
_inputdim(v) = length(v)

"""
    _validate_obs(X, y)

Validate an observation batch before conditioning: equal counts, non-empty, all-finite,
and consistent input dimension. Throws `ArgumentError` with an actionable message.
"""
function _validate_obs(X, y)
    length(X) == length(y) ||
        throw(ArgumentError("observation count mismatch: $(length(X)) inputs vs $(length(y)) values"))
    isempty(X) && throw(ArgumentError("cannot condition a GP on an empty observation set"))
    all(_allfinite, X) || throw(ArgumentError("input set contains non-finite (NaN/Inf) values"))
    all(_allfinite, y) || throw(ArgumentError("observed values contain non-finite (NaN/Inf) values"))
    allequal(_inputdim(x) for x in X) ||
        throw(ArgumentError("inputs have inconsistent dimension: $(unique(_inputdim(x) for x in X))"))
    return nothing
end
```

Then call it at the top of the vector `update` (the scalar method delegates to it). In `src/spine.jl` `update(g::ExactGP, X::AbstractVector, y::AbstractVector)` (line 101), insert as the first line of the body:

```julia
    _validate_obs(X, y)
```

- [ ] **Step 4: Coerce + validate labels in `src/laplace.jl`** — replace `update` (lines 74-79) with:

```julia
function _to_labels(y)
    eltype(y) === Bool && return collect(Bool, y)
    vals = unique(y)
    Set(vals) ⊆ Set((0, 1))  && return Bool[yi == 1 for yi in y]
    Set(vals) ⊆ Set((-1, 1)) && return Bool[yi == 1 for yi in y]
    throw(ArgumentError("LaplaceGP labels must encode two classes as Bool, {0,1}, or {-1,+1}; got value set $(sort(vals))"))
end

function update(g::LaplaceGP, X::AbstractVector, y::AbstractVector)
    yb = _to_labels(y)
    _validate_obs(X, yb)
    xall = vcat(g.x, collect(X))
    yall = vcat(g.y, yb)
    a, W, L = _laplace_fit(g.prior, xall, yall)
    return LaplaceGP(g.prior, xall, yall, a, W, L)
end
```

- [ ] **Step 5: Add `Box` inner constructor + `grid_points` guard in `src/maximize.jl`** — replace the `Box` struct (line 18) with:

```julia
struct Box{T} <: AcquisitionDomain
    lb::T
    ub::T
    function Box(lb::T, ub::T) where {T}
        length(lb) == length(ub) ||
            throw(ArgumentError("Box bounds differ in length: lb has $(length(lb)), ub has $(length(ub))"))
        all(lb .≤ ub) ||
            throw(ArgumentError("Box requires lb .≤ ub; violated at dimension(s) $(findall(lb .> ub))"))
        return new{T}(lb, ub)
    end
end
```

And replace `grid_points` (lines 55-58) body's first lines so the size guard runs before allocation:

```julia
function grid_points(box::Box; per_axis::Int = 50)
    d = length(box.lb)
    big(per_axis)^d > 1_000_000 &&
        throw(ArgumentError("grid of $(per_axis)^$(d) points exceeds 10^6; use SobolPolish() or a Points domain for high-D boxes"))
    axes = [range(box.lb[i], box.ub[i]; length = per_axis) for i in eachindex(box.lb)]
    return [collect(p) for p in Iterators.product(axes...)] |> vec
end
```

- [ ] **Step 6: Validate in `observe!`** — in `src/loop.jl` `observe!` (lines 43-49), insert after the two `_batch` lines (after line 45) and before the `append!`:

```julia
    _validate_obs(X_batch, Y_batch)
```

- [ ] **Step 7: Register the new test file** — in `test/runtests.jl`, add after line 3:

```julia
    include("test_validation.jl")
```

- [ ] **Step 8: Run the new test, then the full suite**

Run: `julia --project=. test/test_validation.jl` → Expected: all PASS.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: full suite green (now 89+ tests).

- [ ] **Step 9: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/spine.jl src/laplace.jl src/maximize.jl src/loop.jl test/test_validation.jl test/runtests.jl
git commit -m "feat(validate): input validation at GP/loop/domain boundaries + label coercion"
```

---

### Task 2: Scale-invariant `classify`

**Files:**
- Modify: `src/saddle.jl` (`classify`, lines 11-18)
- Modify: `test/test_saddle.jl` (add a scale-invariance testset)

**Interfaces:**
- Produces: `classify(H; ε)` where `ε` is now a **relative** tolerance on the eigenvalue spectrum.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test** — append to `test/test_saddle.jl`:

```julia
@testset "classify is scale-invariant" begin
    Hmin = [1.0 0.0; 0.0 2.0]
    @test classify(Hmin) == :min
    @test classify(1.0e-3 .* Hmin) == :min      # absolute-threshold version returned :unclassified here
    @test classify(1.0e3 .* Hmin) == :min
    Hsaddle = [-1.0 0.0; 0.0 2.0]
    @test classify(1.0e-3 .* Hsaddle) == :saddle
    @test classify([0.0 0.0; 0.0 0.0]) == :unclassified   # genuinely flat → undetermined
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_saddle.jl`
Expected: FAIL on `classify(1.0e-3 .* Hmin)` — the absolute `ε=1e-3` floor trips on the `0.001` and `0.002` eigenvalues and returns `:unclassified`.

- [ ] **Step 3: Make the threshold relative** — replace `classify` (lines 11-18) with:

```julia
function classify(H; ε::Real = 1.0e-3)
    λ = eigvals(Symmetric(H))
    scale = maximum(abs, λ)
    scale < eps() && return :unclassified                 # ~flat: curvature undetermined
    any(<(ε * scale), abs.(λ)) && return :unclassified    # eigenvalue negligible *relative* to spectrum
    nneg = count(<(0), λ)
    nneg == 0 && return :min
    nneg == length(λ) && return :max
    return nneg == 1 ? :saddle : :unclassified
end
```

- [ ] **Step 4: Run the test, then the full suite**

Run: `julia --project=. test/test_saddle.jl` → Expected: PASS (including the new testset).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green. (The `test_exemplar_critpoints.jl` survey inlines its own absolute-threshold classify and is unaffected.)

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/saddle.jl test/test_saddle.jl
git commit -m "fix(classify): scale-relative eigenvalue threshold (scale-invariant Morse index)"
```

---

### Task 3: Convergence-gating fields on the stationary-point walkers

Add `converged`/`residual` to `newton_polish`/`saddle_walk` returns and surface them in
`transition_state`. The return becomes a **named tuple whose first three fields stay
`(x, μ∇, H)`**, so existing positional consumers (`x, μ∇, H = saddle_walk(...)`, `p[2]` in
`test_saddle.jl:62`) keep working — the new fields are additive.

**Files:**
- Modify: `src/saddle.jl` (`newton_polish` 30-39, `saddle_walk` 59-70, `transition_state` 104-125)
- Modify: `test/test_saddle.jl` (add a non-convergence testset)

**Interfaces:**
- Produces: `newton_polish(...)`/`saddle_walk(...) -> (; x, μ∇, H, residual, converged)`; `transition_state(...)` result gains `converged`/`residual`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test** — append to `test/test_saddle.jl`:

```julia
@testset "walkers report convergence honestly" begin
    Random.seed!(4)
    # A monotone ramp on the normalised MB box has NO interior critical point: the walker
    # must clamp to a boundary and report converged=false with a non-small residual.
    ramp(p) = p[1]
    X = [MB_BOX.lb .+ (MB_BOX.ub .- MB_BOX.lb) .* rand(2) for _ in 1:20]
    g = Magpie.update(ExactGP(mbkernel(); noise = NOISE), X, ramp.(X))
    r = newton_polish(g, (MB_BOX.lb .+ MB_BOX.ub) ./ 2; box = MB_BOX)
    @test r.converged == false
    @test r.residual > 1.0e-2
    @test r.x == r[1] && r.H == r[3]      # positional compatibility preserved
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_saddle.jl`
Expected: FAIL — `newton_polish` currently returns a 3-tuple, so `r.converged` is a field error.

- [ ] **Step 3: Add the fields to the walkers** — replace `newton_polish` (lines 30-39):

```julia
function newton_polish(g::ExactGP, x0; box::Box, iters::Int = 20, λ::Real = 1.0e-6, tol::Real = 1.0e-8)
    x = collect(float.(x0))
    μ∇, _, H = grad_predict(g, x)
    for _ in 1:iters
        norm(μ∇) < tol && break
        x = clamp.(x .- (Symmetric(H) + λ * I) \ μ∇, box.lb, box.ub)
        μ∇, _, H = grad_predict(g, x)
    end
    res = norm(μ∇)
    return (; x, μ∇, H, residual = res, converged = res < tol)
end
```

and replace `saddle_walk` (lines 59-70):

```julia
function saddle_walk(g::ExactGP, x0; box::Box, iters::Int = 60, η::Real = 0.05, tol::Real = 1.0e-7)
    x = collect(float.(x0))
    for _ in 1:iters
        μ∇, _, H = grad_predict(g, x)
        norm(μ∇) < tol && break
        v = eigen(Symmetric(H)).vectors[:, 1]          # lowest-curvature mode
        F = -μ∇ + 2 * dot(μ∇, v) * v                    # reflect along v: ascend it, descend rest
        x = clamp.(x .+ η .* F, box.lb, box.ub)
    end
    μ∇, _, H = grad_predict(g, x)
    res = norm(μ∇)
    return (; x, μ∇, H, residual = res, converged = res < tol)
end
```

- [ ] **Step 4: Surface convergence in `transition_state`** — in `transition_state` (lines 104-125), the local `predict(gp)` now returns a named tuple. Replace the three destructurings and the return so they use the named fields:

```julia
    history = Tuple{Int, Vector{Float64}, Symbol}[]
    r = predict(g)
    push!(history, (length(X), r.x, classify(r.H)))
    while length(X) < budget
        r = predict(g)
        push!(X, clamp.(r.x, box.lb, box.ub))
        g = buildgp(X)
        r2 = predict(g)
        push!(history, (length(X), r2.x, classify(r2.H)))
    end
    r = predict(g)
    return (saddle = r.x, kind = classify(r.H), converged = r.converged, residual = r.residual, g = g, history = history)
```

- [ ] **Step 5: Run the test, then the full suite**

Run: `julia --project=. test/test_saddle.jl` → Expected: PASS. (The existing `x, μ∇, H = saddle_walk(...)` at line 39 and `p[2]`/`p[1]`/`p[3]` at lines 62-64 still work via positional destructuring/indexing of the named tuple.)
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green.

- [ ] **Step 6: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/saddle.jl test/test_saddle.jl
git commit -m "feat(saddle): walkers report converged/residual; transition_state surfaces them"
```

---

### Task 4: Kernel-generic `fit`

`fit` hard-asserts `SqExponentialKernel`; make it reconstruct the same family for
SqExponential/Matérn-3/2/Matérn-5/2 (the set `grad_predict` supports), and add the missing
Mooncake-vs-finite-difference guard on the two-parameter penalized loss.

**Files:**
- Modify: `src/fit.jl` (`fit`, lines 65-85; add `_kernelfamily`)
- Modify: `test/test_ad.jl` (add the two-parameter penalized-loss AD test)
- Modify: `test/test_fit.jl` (add a Matérn-fit test)

**Interfaces:**
- Produces: `fit(g)` works for SqExponential/Matern32/Matern52 base kernels; throws `ArgumentError` otherwise. `Magpie._kernelfamily(basekernel) -> Kernel`.
- Consumes: existing `_basekernel`/`_outputscale`/`_lengthscale` (already in `fit.jl`).

- [ ] **Step 1: Write the failing tests** — append to `test/test_fit.jl`:

```julia
@testset "fit is kernel-generic over supported families" begin
    X = [randn(2) for _ in 1:40]; y = [sum(abs2, xi) for xi in X]
    g52 = Magpie.update(ExactGP(with_lengthscale(Matern52Kernel(), 0.7); noise = 1e-4), X, y)
    fitted = Magpie.fit(g52)
    @test Magpie._basekernel(fitted.prior.kernel) isa Matern52Kernel    # family preserved, not swapped to RBF
    @test Magpie.nlml(fitted) ≤ Magpie.nlml(g52)                         # fit improved (or matched) the objective
    glin = Magpie.update(ExactGP(LinearKernel(); noise = 1e-4), X, y)    # unsupported family
    @test_throws ArgumentError Magpie.fit(glin)
end
```

and append the AD guard to `test/test_ad.jl` (before the final `end` of the file — it is a top-level `@testset`, so add a sibling testset):

```julia
@testset "two-parameter penalized fit loss: Mooncake == finite diff" begin
    X = [randn(2) for _ in 1:12]; y = [sum(abs2, xi) for xi in X]
    μ0, σ0 = 0.0, 0.75                                       # the :auto prior centre/width
    loss(p) = nlml(Magpie.update(ExactGP(exp(p[2]) * with_lengthscale(SqExponentialKernel(), exp(p[1])); noise = 1e-6), X, y)) +
              0.5 * ((p[1] - μ0) / σ0)^2
    p = [0.3, 0.1]
    g_mc = DifferentiationInterface.gradient(loss, AutoMooncake(; config = nothing), p)
    h = 1e-6
    g_fd = [(loss(p .+ h .* (1:2 .== i)) - loss(p .- h .* (1:2 .== i))) / 2h for i in 1:2]
    @test g_mc ≈ g_fd rtol=1e-4
end
```

- [ ] **Step 2: Run them to verify they fail**

Run: `julia --project=. test/test_fit.jl` → Expected: FAIL — `fit(g52)` hits `@assert ... isa SqExponentialKernel`.
Run: `julia --project=. test/test_ad.jl` → Expected: PASS already if the closure differentiates, but it is a NEW guard; if it errors, that is the regression it exists to catch. (It should pass — this locks the contract.)

- [ ] **Step 3: Make `fit` kernel-generic** — in `src/fit.jl`, add after the peelers (line 11):

```julia
_kernelfamily(::SqExponentialKernel) = SqExponentialKernel()
_kernelfamily(::Matern32Kernel) = Matern32Kernel()
_kernelfamily(::Matern52Kernel) = Matern52Kernel()
_kernelfamily(k) = throw(ArgumentError("fit supports SqExponential/Matern32/Matern52 base kernels; got $(typeof(k)). (grad_predict's derivative path covers the same set.)"))
```

Then in `fit` (lines 65-85): delete the `@assert` line (66), and replace the `mkkernel` definition (line 74) with a family-reconstructing version:

```julia
function fit(g::ExactGP; restarts::Int = 1, ad = AutoForwardDiff(), ℓ_prior = :auto)
    fam = _kernelfamily(_basekernel(g.prior.kernel))     # validates + returns a fresh base kernel of the same family
    X = g.x; y = g.δ .+ AbstractGPs.mean(g.prior, g.x)
    noise = g.noise; meanfn = g.prior.mean
    logℓ0 = log(_lengthscale(g.prior.kernel))
    p0 = [logℓ0, log(_outputscale(g.prior.kernel))]
    pri = ℓ_prior === :auto ? (logℓ0, 0.75) : ℓ_prior
    penalty(p) = pri === nothing ? zero(eltype(p)) : 0.5 * ((p[1] - pri[1]) / pri[2])^2
    mkkernel(p) = exp(p[2]) * with_lengthscale(fam, exp(p[1]))
    loss(p, _) = nlml(update(ExactGP(mkkernel(p); noise = noise, mean = meanfn), X, y)) + penalty(p)
    obj(p) = loss(p, nothing)
    best = g; best_obj = obj(p0)
    for r in 1:restarts
        start = r == 1 ? p0 : p0 .+ 0.1 .* randn(2)
        prob = OptimizationProblem(OptimizationFunction(loss, ad), start; lb = [-6.0, -6.0], ub = [6.0, 6.0])
        sol = solve(prob, LBFGS())
        obj(sol.u) < best_obj && ((best, best_obj) = (update(ExactGP(mkkernel(sol.u); noise = noise, mean = meanfn), X, y), obj(sol.u)))
    end
    return best
end
```

Also update the `!!! note` in the docstring (lines 62-64) to say `fit` supports SqExponential/Matérn-3/2/Matérn-5/2.

- [ ] **Step 4: Run both tests, then the full suite**

Run: `julia --project=. test/test_fit.jl` and `julia --project=. test/test_ad.jl` → Expected: PASS.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green.

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/fit.jl test/test_fit.jl test/test_ad.jl
git commit -m "feat(fit): kernel-generic over SqExp/Matern32/Matern52 + Mooncake AD guard on penalized loss"
```

---

### Task 5: Thread an RNG through the randomized acquisitions and saddle seeding

Make `resample(a, rng)` the primary method and give `_ts_seed`/`transition_state` an `rng`
keyword, so randomness is injected rather than ambient. (The loop-side rng field is Task 6,
which depends on `resample(a, rng)` existing.)

**Files:**
- Modify: `src/acquisitions.jl` (`resample`, lines 72-73, 113, 153)
- Modify: `src/saddle.jl` (`_ts_seed` 74-83, `transition_state` 104-125)
- Modify: `test/test_saddle.jl` (reproducibility testset)

**Interfaces:**
- Produces: `resample(a::AcquisitionFunction, rng) -> AcquisitionFunction` (primary); `transition_state(...; rng=Random.default_rng())`.
- Consumes: nothing. Task 6 consumes `resample(a, rng)`.

- [ ] **Step 1: Write the failing test** — append to `test/test_saddle.jl`:

```julia
@testset "transition_state is reproducible under a seeded rng" begin
    r1 = transition_state(mbt, MB_min, MC; kernel = mbkernel(), noise = NOISE, box = MB_BOX,
                          budget = 10, nseed = 5, predictor = :minmode, rng = MersenneTwister(123))
    r2 = transition_state(mbt, MB_min, MC; kernel = mbkernel(), noise = NOISE, box = MB_BOX,
                          budget = 10, nseed = 5, predictor = :minmode, rng = MersenneTwister(123))
    @test r1.saddle == r2.saddle
end
```

(`MersenneTwister`/`Random` are already imported at the top of `test_saddle.jl`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_saddle.jl`
Expected: FAIL — `transition_state` has no `rng` keyword (`MethodError`/unsupported kwarg).

- [ ] **Step 3: Add `resample(a, rng)` primary methods** — in `src/acquisitions.jl`, replace the three `resample` definitions (lines 72-73, 113, 153) with the two-arg primaries plus single-arg delegates:

```julia
resample(a::AcquisitionFunction, rng) = a
resample(a::RandStraddle, rng) = RandStraddle(a.h, sqrt(-2 * log(rand(rng))), rng)
resample(a::RandGradStraddle, rng) = RandGradStraddle(sqrt(-2 * log(rand(rng))), rng)
resample(a::LocalPenalization, rng) = LocalPenalization(resample(a.base, rng), a.pts, a.c, a.s)

resample(a::AcquisitionFunction) = a
resample(a::RandStraddle) = resample(a, a.rng)
resample(a::RandGradStraddle) = resample(a, a.rng)
resample(a::LocalPenalization) = LocalPenalization(resample(a.base), a.pts, a.c, a.s)
```

(Place the `resample(a::RandGradStraddle, ...)` near `RandGradStraddle` and the `LocalPenalization` ones near `LocalPenalization` if you prefer; co-location is optional. The single-arg `resample(a::RandStraddle)` etc. keep standalone callers working.)

- [ ] **Step 4: Thread rng through saddle seeding** — in `src/saddle.jl`, replace `_ts_seed` (lines 74-83):

```julia
function _ts_seed(m1, m2, box; nseed::Int = 5, jit::Real = 0.12, rng = Random.default_rng())
    d = m2 .- m1
    perp = [-d[2], d[1]]; perp = perp ./ max(norm(perp), 1.0e-9)
    pts = [collect(float.(m1)), collect(float.(m2))]
    for i in 1:nseed
        t = i / (nseed + 1); base = m1 .+ t .* d
        push!(pts, clamp.(base .+ (jit * (2 * rand(rng) - 1)) .* perp, box.lb, box.ub))
    end
    return pts
end
```

and add the `rng` keyword to `transition_state` (line 104-105 signature) and pass it to `_ts_seed`:

```julia
function transition_state(f, m1, m2; kernel, noise::Real = 1.0e-3, box::Box, budget::Int = 12,
                          nseed::Int = 5, predictor::Symbol = :minmode, η::Real = 0.05,
                          rng = Random.default_rng())
    m1 = collect(float.(m1)); m2 = collect(float.(m2))
    X = _ts_seed(m1, m2, box; nseed = nseed, rng = rng)
```

(the rest of the body is unchanged from Task 3.) Ensure `using Random` is available in `saddle.jl` — add `using Random: default_rng` at the top if not already imported via the module.

- [ ] **Step 5: Run the test, then the full suite**

Run: `julia --project=. test/test_saddle.jl` → Expected: PASS (the existing `Random.seed!`-based tests still pass because the default rng path is unchanged).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green.

- [ ] **Step 6: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/acquisitions.jl src/saddle.jl test/test_saddle.jl
git commit -m "feat(rng): resample(a, rng) primary + seeded transition_state for reproducible runs"
```

---

### Task 6: Type-stable, seedable `ActiveLearner`

Parameterize storage on the data (`{TX,TY}`), keep `gp`/`acq` abstract (so in-place
reassignment across type changes still works), add an `rng` field, and thread it through
`run!`/`acquire`.

**Files:**
- Modify: `src/loop.jl` (`ActiveLearner` struct + constructors 21-25; `acquire` 64-67; `run!` 77-87)
- Modify: `test/test_loop.jl` (type-stability + reproducibility tests)

**Interfaces:**
- Consumes: `resample(a, rng)` from Task 5.
- Produces: `ActiveLearner{TX,TY}`; `ActiveLearner(gp, acq; rng=Random.default_rng())` (inferred defaults); `run!(al, oracle; …)` reproducible under `al.rng`.

- [ ] **Step 1: Write the failing test** — append to `test/test_loop.jl` (`Random` is imported at the top):

```julia
@testset "ActiveLearner has typed storage and is seedable" begin
    mk() = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1e-4),
                         Magpie.RandStraddle(); rng = MersenneTwister(42))
    al = mk()
    f(x) = sum(x)^2 - 1
    observe!(al, [0.0], f([0.0]))
    @test eltype(al.Xs) != Any                      # concrete input storage
    @test eltype(al.Ys) != Any                      # concrete value storage
    a1 = run!(mk(), f; budget = 8, over = Box([-1.0], [1.0]))
    a2 = run!(mk(), f; budget = 8, over = Box([-1.0], [1.0]))
    @test queried_points(a1) == queried_points(a2)  # same seed → identical survey
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_loop.jl`
Expected: FAIL — `ActiveLearner(...; rng=...)` has no `rng` keyword, and `eltype(al.Xs)` is `Any`.

- [ ] **Step 3: Parameterize the struct + constructors** — replace `src/loop.jl` lines 21-25 with:

```julia
mutable struct ActiveLearner{TX, TY}
    gp::AbstractGPModel              # abstract on purpose: concrete type changes on first update
    acq::AcquisitionFunction         # abstract on purpose: changes when wrapped by LocalPenalization
    Xs::Vector{TX}
    Ys::Vector{TY}
    acq_vals::Vector{Float64}
    rng::AbstractRNG
end

# Observation element type implied by the GP kind (extended for multi-output in Phase 1.5).
_obs_eltype(::ExactGP) = Float64
_obs_eltype(::LaplaceGP) = Bool

# Typed escape hatch: caller fixes the input/value element types.
ActiveLearner{TX, TY}(gp, acq; rng = Random.default_rng()) where {TX, TY} =
    ActiveLearner{TX, TY}(gp, acq, TX[], TY[], Float64[], rng)

# Convenience: infer Vector{Float64} inputs and the value type from the GP.
ActiveLearner(gp, acq; rng = Random.default_rng()) =
    ActiveLearner{Vector{Float64}, _obs_eltype(gp)}(gp, acq; rng = rng)
```

- [ ] **Step 4: Thread rng through `acquire` and `run!`** — replace `acquire` (lines 64-67):

```julia
function acquire(al::ActiveLearner; over, maximizer = default_for(over), q::Int = 1)
    q == 1 || error("batch acquisition (q>1) not yet implemented; use q=1")
    return acquire(al.gp, resample(al.acq, al.rng); over = over, maximizer = maximizer)
end
```

and `run!` (lines 77-87):

```julia
function run!(al::ActiveLearner, oracle; budget::Int, over, stop = al -> false, refit_every::Int = 0)
    for t in 1:budget
        stop(al) && break
        acq = resample(al.acq, al.rng)
        x = acquire(al.gp, acq; over = over)
        push!(al.acq_vals, acq(al.gp, x))
        observe!(al, x, oracle(x))
        refit_every > 0 && t % refit_every == 0 && fit!(al)
    end
    return al
end
```

- [ ] **Step 5: Run the test, then the full suite**

Run: `julia --project=. test/test_loop.jl` → Expected: PASS. The existing `ActiveLearner(gp, Straddle(...))` call sites in `test_loop.jl`/`test_exemplar_critpoints.jl` still construct via the inferred-default convenience constructor; `al.acq = LocalPenalization(al.acq, al.Xs; …)` still works (abstract `acq` field).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green.

- [ ] **Step 6: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/loop.jl test/test_loop.jl
git commit -m "feat(loop): typed ActiveLearner{TX,TY} + seedable rng threaded through run!/acquire"
```

---

### Task 7: Reconcile the superseded critical-point exemplar

The second testset of `test/test_exemplar_critpoints.jl` asserts an "active learning beats
random for enumeration" advantage the design spec explicitly retracts. Rewrite it to assert
only extraction quality (the honest claim); the genuine active-learning win lives in
`test_saddle.jl`'s `transition_state` test.

**Files:**
- Modify: `test/test_exemplar_critpoints.jl` (replace the testset at lines 62-100)

**Interfaces:**
- Consumes: the file-local `critical_points` helper (lines 37-49), unchanged.
- Produces: nothing public.

- [ ] **Step 1: Replace the superseded testset** — delete lines 62-100 (the `@testset "Active learning recovers localized critical points where random fails"`) and replace with:

```julia
@testset "critical_points recovers the 9 critical points of a multi-well surface" begin
    # f = (x²-1)² + (y²-1)² has 9 critical points: 4 minima (±1,±1), 1 max (0,0),
    # 4 saddles (±1,0),(0,±1). This asserts EXTRACTION quality on a well-sampled GP.
    # The earlier "active beats random for enumeration" framing was retracted as an extraction
    # artifact (see the critical-point design spec); the genuine active-learning advantage is the
    # TARGETED transition_state search exercised in test_saddle.jl, not enumeration.
    f(x) = (x[1]^2 - 1)^2 + (x[2]^2 - 1)^2
    box = Box([-2.0, -2.0], [2.0, 2.0])
    want = vcat([(:min, [a, b]) for a in (-1.0, 1.0) for b in (-1.0, 1.0)],
                [(:max, [0.0, 0.0])],
                [(:saddle, s) for s in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0])])
    recovered(cps) = count(((k, p),) -> any(c -> c.kind == k && isapprox(c.point, p; atol = 0.25), cps), want)

    Random.seed!(1)
    X = [4 .* rand(2) .- 2 for _ in 1:120]
    g = Magpie.fit(update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.6); noise = 1e-4), X, f.(X)))
    @test recovered(critical_points(g, box)) ≥ 7      # extraction recovers most of the 9
end
```

- [ ] **Step 2: Run the file**

Run: `julia --project=. test/test_exemplar_critpoints.jl`
Expected: PASS — both testsets (the quadratic bowl and the rewritten multi-well survey). If `recovered(...)` falls below 7, raise the sample count (`1:120 → 1:180`) or widen `atol` slightly to `0.3`; the surface is densely sampled on `[-2,2]²` so recovery should be high.

- [ ] **Step 3: Full suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green, with no test asserting active-beats-random for enumeration anywhere.

- [ ] **Step 4: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add test/test_exemplar_critpoints.jl
git commit -m "test(critpoints): rewrite superseded exemplar to honest extraction-quality assertion"
```

---

## Self-Review

**Spec coverage** (Phase 1 workstreams → tasks):
- 1.1 Input validation → **Task 1** ✓ (`_validate_obs`, Box ctor, grid guard, D2 labels)
- 1.2 RNG threading → **Task 5** (acquisitions/saddle) + **Task 6** (loop rng field) ✓
- 1.3 Type-stable storage (D1) → **Task 6** ✓ (data-only `{TX,TY}`, abstract `gp`/`acq`)
- 1.4 Convergence gating (D3) → **Task 3** ✓ (additive `converged`/`residual`)
- 1.5 Scale-invariant classify → **Task 2** ✓
- 1.6 Kernel-generic fit (D4) → **Task 4** ✓ (+ the spec's missing Mooncake AD test)
- 1.7 Reconcile exemplar → **Task 7** ✓
- Phase 1.5 (multi-output spine) → **deferred to its own plan** (noted below), per the spec's phase split. `_obs_eltype` (Task 6) gets a one-line `ExactGP` extension there.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every run step states the command and expected result.

**Type consistency:** `_validate_obs(X, y)` signature is identical across spine/laplace/loop call sites. The walker named tuple `(; x, μ∇, H, residual, converged)` keeps `(x, μ∇, H)` as fields 1-3 for positional compatibility (Task 3), and `transition_state` consumes `.x`/`.H`/`.converged`/`.residual` consistently. `resample(a, rng)` (Task 5) is the exact method `run!`/`acquire` call (Task 6). `_obs_eltype` returns `Float64`/`Bool` matching `TY`.

## Follow-on (separate plans)

- **Phase 1.5 — Multi-output spine:** generalize `ExactGP` to `n×d` weights aligned to gp-ude `ExactGPField` (its own plan; extends `_obs_eltype(::ExactGP)` to `g.d == 1 ? Float64 : Vector{Float64}`).
- **Phase 2 — Public API & ergonomics;** **Phase 3 — Classification parity.** Per the design spec.
