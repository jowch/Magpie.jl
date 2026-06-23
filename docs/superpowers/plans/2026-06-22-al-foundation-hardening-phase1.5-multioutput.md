# Phase 1.5 — Multi-Output Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the active-learning `ExactGP` from single-output to `d` independent outputs sharing one kernel, converging on the gp-ude `ExactGPField` representation, without changing the `d=1` path.

**Architecture:** Store `δ`/`α` as a `Vector` when `d=1` (unchanged) and an `n×d` `Matrix` when `d>1`. The shared-kernel Cholesky `C` is identical to the single-output case (only the right-hand side gains columns), so `update_chol`/`_chol`/the AD story carry over untouched. Posterior mean becomes per-output (length-`d` vector / `nx×d` matrix); **posterior variance is shared across outputs** (computed once). Multi-output *acquisitions* and *fitting* are out of scope — scalar-only paths (`grad_predict`, `fit`, acquisitions) error clearly on `d>1`.

**Tech Stack:** Julia ≥1.10, AbstractGPs.jl (`update_chol`, `diag_Xt_invA_X`, `Xt_invA_X`, `Xt_invA_Y`), KernelFunctions.jl, LinearAlgebra.

Source of truth: the **Phase 1.5** section of [the design spec](../specs/2026-06-22-al-foundation-hardening-design.md).

## Global Constraints

- **All factorizations through `_chol`** (`cholesky(Symmetric(K); check=false)`), unchanged. `C` is shared across outputs — do not introduce a per-output factorization.
- **The `d=1` path must stay byte-for-byte equivalent** — `δ`/`α` remain `Vector`s for `d=1`; the full existing suite (118 tests) stays green.
- **Posterior variance is shared across outputs** — `var`/`mean_and_var`'s variance term is a single per-point vector regardless of `d` (independent outputs, shared kernel ⇒ identical marginal variance). Do not compute it per-output.
- **Representation aligns with gp-ude `ExactGPField`:** an output-dimension field `d::Int`, weights as an `n×d` matrix, shared `_chol`.
- **Multi-output acquisitions and fitting are deferred** — `grad_predict`, `fit`, and `acquire` raise a clear `ArgumentError` on `d>1`.
- **Julia ≥1.10**; format with Runic before each commit; run the full suite before each commit.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/spine.jl` | `ExactGP` `d` field + `_obsmatrix`; matrix-aware `update`/`_update_incremental`; multi-output `mean`/`mean_and_var`; `predmean` guard | 1, 2 |
| `src/fit.jl` | `nlml` multi-output correction; `fit` `d>1` guard | 1, 3 |
| `src/derivatives.jl` | `grad_predict` `d>1` guard | 3 |
| `src/loop.jl` | `_obs_eltype` d-aware | 3 |
| `src/maximize.jl` | `_outputdim` + `acquire` `d>1` guard | 3 |
| `test/test_multioutput.jl` | NEW — all multi-output tests | 1, 2, 3 |
| `test/runtests.jl` | register the new test file | 1 |

---

### Task 1: Multi-output representation — struct, conditioning, marginal likelihood

**Files:**
- Modify: `src/spine.jl` — `ExactGP` struct (line 46-48), constructor (49-50), add `_obsmatrix`; `update` (123-132); `_update_incremental` (173-181)
- Modify: `src/fit.jl` — `nlml` (lines 26-30)
- Create: `test/test_multioutput.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `ExactGP(kernel; noise, mean, d=1)` (gains `d::Int` field, last position); `update` stores `δ`/`α` as `n×d` matrices when `d>1`, vectors when `d=1`; `Magpie._obsmatrix(y, d)`; `nlml` multi-output-correct.
- Consumes: nothing from later tasks.

- [ ] **Step 1: Write the failing test** — create `test/test_multioutput.jl`:

```julia
using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, update, nlml

# 2-output target; condition a GP and check the representation + conditioning.
f2(x) = [x[1] + x[2], x[1] - x[2]]
mk(d) = ExactGP(with_lengthscale(SqExponentialKernel(), 0.7); noise = 1e-4, d = d)

@testset "multi-output conditioning: n×d weights, shared Cholesky" begin
    Random.seed!(3)
    X = [randn(2) for _ in 1:8]
    Y = [f2(x) for x in X]                         # vector of length-2 observations
    g = update(mk(2), X, Y)
    @test g.d == 2
    @test size(g.α) == (8, 2)                       # weights are n×d (gp-ude ExactGPField layout)
    @test size(g.δ) == (8, 2)
end

@testset "incremental update == from-scratch (d=2)" begin
    Random.seed!(4)
    X = [randn(2) for _ in 1:6]; Y = [f2(x) for x in X]
    g_scratch = update(mk(2), X, Y)
    g_incr = update(update(mk(2), X[1:3], Y[1:3]), X[4:6], Y[4:6])
    xs = [randn(2) for _ in 1:4]
    @test mean(g_scratch, xs) ≈ mean(g_incr, xs) rtol=1e-9
end

@testset "d=1 nlml unchanged (regression guard)" begin
    Random.seed!(5)
    X = [randn(2) for _ in 1:10]; y = [sum(x) for x in X]
    g = update(mk(1), X, y)
    n = length(y)
    expected = 0.5 * dot(g.δ, g.α) + sum(log, diag(g.C.U)) + 0.5n * log(2π)
    @test nlml(g) ≈ expected rtol=1e-12
end

@testset "_obsmatrix validates observation length" begin
    @test_throws ArgumentError Magpie._obsmatrix([[1.0, 2.0], [3.0]], 2)   # ragged
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_multioutput.jl`
Expected: FAIL — `ExactGP(...; d=2)` has no `d` keyword; `_obsmatrix` undefined.

- [ ] **Step 3: Add the `d` field, constructor, and `_obsmatrix`** — in `src/spine.jl`, replace the struct + constructor (lines 46-50):

```julia
struct ExactGP{Tp, Tx, Tδ, TC, Tα} <: AbstractGPModel
    prior::Tp; x::Tx; δ::Tδ; C::TC; α::Tα; noise::Float64; d::Int
end
ExactGP(kernel::Kernel; noise::Real = 1.0e-6, mean = AbstractGPs.ZeroMean(), d::Int = 1) =
    ExactGP(AbstractGPs.GP(mean, kernel), Any[], Float64[], nothing, Float64[], Float64(noise), d)
```

and add `_obsmatrix` just above `update` (after `_validate_obs`, ~line 113):

```julia
# Observation values → the conditioning RHS. d=1 keeps a Vector (the single-output path is
# unchanged); d>1 stacks the per-point length-d observations into an n×d Matrix.
function _obsmatrix(y, d::Int)
    d == 1 && return collect(float.(y))
    all(yi -> length(yi) == d, y) ||
        throw(ArgumentError("each observation must have length d=$d; got lengths $(unique(length.(y)))"))
    return permutedims(reduce(hcat, [collect(float.(yi)) for yi in y]))    # n×d
end
```

- [ ] **Step 4: Make `update` and `_update_incremental` matrix-aware** — replace `update` (lines 123-132):

```julia
function update(g::ExactGP, X::AbstractVector, y::AbstractVector)
    _validate_obs(X, y)
    _hasdata(g) && return _update_incremental(g, X, y)
    xnew = collect(X)
    δnew = _obsmatrix(y, g.d) .- AbstractGPs.mean(g.prior, xnew)     # vector (d=1) or n×d matrix
    K = AbstractGPs.cov(g.prior, xnew) + g.noise * I
    C = _chol(K)
    return ExactGP(g.prior, xnew, δnew, C, C \ δnew, g.noise, g.d)
end
update(g::ExactGP, x, y::Real) = update(g, [x], [y])
```

and `_update_incremental` (lines 173-181):

```julia
function _update_incremental(g::ExactGP, X::AbstractVector, y::AbstractVector)
    xnew = collect(X)
    C12 = AbstractGPs.cov(g.prior, g.x, xnew)
    C22 = Matrix(Symmetric(AbstractGPs.cov(g.prior, xnew) + g.noise * I))
    Cext = update_chol(g.C, C12, C22)
    xall = vcat(g.x, xnew)
    δall = vcat(g.δ, _obsmatrix(y, g.d) .- AbstractGPs.mean(g.prior, xnew))   # vcat rows; matrix-safe
    return ExactGP(g.prior, xall, δall, Cext, Cext \ δall, g.noise, g.d)
end
```

(`vcat` of two `Vector`s gives a `Vector` (d=1, unchanged); `vcat` of two `n×d`/`m×d` matrices stacks rows. `Cext \ δall` is matrix-vector for d=1, matrix-matrix for d>1 — both correct.)

- [ ] **Step 5: Correct `nlml` for multi-output** — in `src/fit.jl`, replace `nlml` (lines 26-30):

```julia
function nlml(g::ExactGP)
    _hasdata(g) || return 0.0
    n = size(g.δ, 1)                                   # rows = #points (length for a Vector)
    return 0.5 * dot(g.δ, g.α) + g.d * (sum(log, diag(g.C.U)) + 0.5n * log(2π))
end
```

(For `d=1`: `size(δ,1)=length(δ)`, `dot` is the vector dot, `g.d=1` ⇒ identical to the old formula. For `d>1`: `dot(δ,α)=Σ_d δ_d'α_d`, and the shared-`C` log-det + 2π terms are counted once per output.)

- [ ] **Step 6: Register the test file** — in `test/runtests.jl`, add after the `test_saddle.jl` include:

```julia
    include("test_multioutput.jl")
```

- [ ] **Step 7: Run the new test, then the full suite**

Run: `julia --project=. test/test_multioutput.jl` → Expected: PASS (the d=1 and d=2 testsets in scope so far; `mean` already works for d>1 via dispatch).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green (d=1 unchanged).

- [ ] **Step 8: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/spine.jl src/fit.jl test/test_multioutput.jl test/runtests.jl
git commit -m "feat(spine): multi-output ExactGP — n×d weights, shared Cholesky, nlml correction"
```

---

### Task 2: Multi-output prediction — mean, variance, predmean

`mean`/`mean_and_var`'s *conditioned* path already returns the right shape for `d>1` via dispatch (`cov * α` is `nx×d`). This task fixes the *unconditioned* branches (which must return `nx×d` for `d>1`), guards `predmean` (it returns a scalar), and adds the multi-output prediction tests — including the shared-variance invariant.

**Files:**
- Modify: `src/spine.jl` — `mean` (69-72), `mean_and_var` (140-145), `predmean` (161-163); `predict` docstring (147-153)
- Modify: `test/test_multioutput.jl`

**Interfaces:**
- Consumes: the `d` field + matrix `α` from Task 1.
- Produces: `mean(g, xs)` returns `nx×d` for `d>1`; `var`/`mean_and_var` variance is the shared per-point vector; `predmean` throws on `d>1`.

- [ ] **Step 1: Write the failing test** — append to `test/test_multioutput.jl`:

```julia
@testset "multi-output prediction shapes + shared variance" begin
    Random.seed!(6)
    X = [randn(2) for _ in 1:8]; Y = [f2(x) for x in X]
    g2 = update(mk(2), X, Y)
    g1 = update(mk(1), X, [y[1] for y in Y])         # same X, same kernel/noise → same C
    xs = [randn(2) for _ in 1:5]

    M = mean(g2, xs)
    @test size(M) == (5, 2)                           # per-output posterior mean
    @test var(g2, xs) ≈ var(g1, xs) rtol=1e-10        # variance shared across outputs (depends only on X)

    Mv, V = mean_and_var(g2, xs)
    @test Mv ≈ M rtol=1e-10
    @test V ≈ var(g2, xs) rtol=1e-10

    # unconditioned multi-output prior mean is nx×d
    @test size(mean(mk(2), xs)) == (5, 2)
end

@testset "predmean is single-output only" begin
    Random.seed!(7)
    X = [randn(2) for _ in 1:6]
    g2 = update(mk(2), X, [f2(x) for x in X])
    @test_throws ArgumentError Magpie.predmean(g2, randn(2))
    g1 = update(mk(1), X, [sum(x) for x in X])
    @test Magpie.predmean(g1, X[1]) ≈ mean(g1, [X[1]])[1] rtol=1e-12   # d=1 still scalar, matches mean
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_multioutput.jl`
Expected: FAIL — `mean(mk(2), xs)` (unconditioned) returns a length-5 vector, not `5×2`; `predmean(g2, …)` does not throw (it hits `dot(vector, matrix)` and errors with a `DimensionMismatch`, not the intended `ArgumentError`).

- [ ] **Step 3: Fix the unconditioned `mean`/`mean_and_var` branches and guard `predmean`** — in `src/spine.jl`:

`mean` (lines 69-72):

```julia
function Statistics.mean(g::ExactGP, xs::AbstractVector)
    m = AbstractGPs.mean(g.prior, xs)
    _hasdata(g) || return g.d == 1 ? m : repeat(m, 1, g.d)
    return m .+ AbstractGPs.cov(g.prior, xs, g.x) * g.α          # vector (d=1) or nx×d (d>1)
end
```

`mean_and_var` (lines 140-145):

```julia
function mean_and_var(g::ExactGP, xs::AbstractVector)
    m = AbstractGPs.mean(g.prior, xs)
    _hasdata(g) || return (g.d == 1 ? m : repeat(m, 1, g.d), AbstractGPs.var(g.prior, xs))
    Ks = AbstractGPs.cov(g.prior, g.x, xs)
    return (m .+ Ks' * g.α, AbstractGPs.var(g.prior, xs) .- diag_Xt_invA_X(g.C, Ks))   # var shared across outputs
end
```

`predmean` (lines 161-163):

```julia
function predmean(g::ExactGP, u)
    g.d == 1 ||
        throw(ArgumentError("predmean returns a scalar but this GP has d=$(g.d) outputs; use mean(g, [u]) for the length-d vector"))
    return _hasdata(g) ?
        only(AbstractGPs.mean(g.prior, [u])) + dot(AbstractGPs.cov(g.prior, g.x, [u]), g.α) :
        only(AbstractGPs.mean(g.prior, [u]))
end
```

Also update the `predict` docstring (lines 147-153) to note that for a `d>1` GP the mean is `nx×d` (per output) and the variance is a length-`nx` vector shared across outputs.

- [ ] **Step 4: Run the test, then the full suite**

Run: `julia --project=. test/test_multioutput.jl` → Expected: PASS.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green (d=1 mean/var/predmean unchanged).

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/spine.jl test/test_multioutput.jl
git commit -m "feat(spine): multi-output mean (per-output) + shared variance; predmean guards d>1"
```

---

### Task 3: Guards and loop wiring for the scalar-only paths

Multi-output acquisitions and fitting are deferred. Make the scalar-only entry points fail clearly on `d>1` rather than silently using output 1, and teach `ActiveLearner` the multi-output value type.

**Files:**
- Modify: `src/derivatives.jl` — `grad_predict` (entry, ~line 22)
- Modify: `src/fit.jl` — `fit` (entry, ~line 65)
- Modify: `src/maximize.jl` — add `_outputdim`; guard `acquire` (line 71)
- Modify: `src/loop.jl` — `_obs_eltype(::ExactGP)` (the method added in Phase 1 Task 6)
- Modify: `test/test_multioutput.jl`

**Interfaces:**
- Consumes: the `d` field from Task 1.
- Produces: `grad_predict`/`fit`/`acquire` throw `ArgumentError` on `d>1`; `Magpie._outputdim(g) -> Int`; `ActiveLearner` over a `d>1` `ExactGP` stores `Vector{Float64}` values.

- [ ] **Step 1: Write the failing test** — append to `test/test_multioutput.jl`:

```julia
using Magpie: grad_predict, fit, acquire, Box, ActiveLearner, Straddle, observe!

@testset "scalar-only paths reject d>1" begin
    Random.seed!(8)
    X = [randn(2) for _ in 1:6]
    g2 = update(mk(2), X, [f2(x) for x in X])
    @test_throws ArgumentError grad_predict(g2, randn(2))
    @test_throws ArgumentError fit(g2)
    @test_throws ArgumentError acquire(g2, Straddle(); over = Box([-2.0, -2.0], [2.0, 2.0]))
end

@testset "ActiveLearner infers multi-output value storage" begin
    al = ActiveLearner(mk(2), Straddle())
    @test eltype(al.Ys) == Vector{Float64}             # d=2 → vector-valued observations
    al1 = ActiveLearner(mk(1), Straddle())
    @test eltype(al1.Ys) == Float64                    # d=1 unchanged
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project=. test/test_multioutput.jl`
Expected: FAIL — `grad_predict`/`fit`/`acquire` don't yet guard `d`; `eltype(al.Ys)` is `Float64` for the `d=2` learner.

- [ ] **Step 3: Add the guards** —

`src/derivatives.jl`, at the top of `grad_predict` (after `function grad_predict(g::ExactGP, x::AbstractVector; hessian::Bool=true)`):

```julia
    g.d == 1 ||
        throw(ArgumentError("grad_predict is single-output (d=1); got d=$(g.d). Multi-output derivatives are not supported."))
```

`src/fit.jl`, at the top of `fit` (after the `function fit(g::ExactGP; …)` line, before the `_kernelfamily` call):

```julia
    g.d == 1 ||
        throw(ArgumentError("fit currently supports single-output GPs (d=1); got d=$(g.d)."))
```

`src/maximize.jl`, add near `acquire` (after line 71's `acquire` definition) an output-dim accessor and guard the entry:

```julia
_outputdim(g) = 1
_outputdim(g::ExactGP) = g.d
```

and change `acquire` (line 71) to guard first:

```julia
function acquire(g, a; over, maximizer = default_for(over))
    _outputdim(g) == 1 ||
        throw(ArgumentError("acquisitions are single-output; got a GP with d=$(_outputdim(g)) outputs."))
    return _acquire(g, a, over, maximizer)
end
```

(`_outputdim` defaults to 1 so `LaplaceGP`/other models are unaffected.)

`src/loop.jl`, change the `_obs_eltype(::ExactGP)` method (added in Phase 1 Task 6) to be d-aware:

```julia
_obs_eltype(g::ExactGP) = g.d == 1 ? Float64 : Vector{Float64}
```

- [ ] **Step 4: Run the test, then the full suite**

Run: `julia --project=. test/test_multioutput.jl` → Expected: PASS.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green. (The `_obs_eltype` change is instance-based; the Phase-1 `ActiveLearner(gp, acq)` convenience constructor already passes the GP instance, and `d=1` GPs still infer `Float64`.)

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add src/derivatives.jl src/fit.jl src/maximize.jl src/loop.jl test/test_multioutput.jl
git commit -m "feat(spine): guard scalar-only paths (grad_predict/fit/acquire) on d>1; d-aware ActiveLearner storage"
```

---

## Self-Review

**Spec coverage** (Phase 1.5 design section → tasks):
- `ExactGP` gains `d`; `δ`/`α` widen to `n×d`; `C`/`_chol`/`update_chol` unchanged → **Task 1** ✓
- `predmean` returns length-`d` (here: scalar for d=1, guarded for d>1 since the scalar contract is kept and multi-output mean uses `mean`) → **Task 2** ✓
- Predictive variance shared across outputs (computed once) → **Task 2** ✓ (test asserts `var(g2)≈var(g1)`)
- `_validate_obs` accepts vector-valued `y` → already true from Phase 1 (`_allfinite` handles vectors); `_obsmatrix` adds the length check → **Task 1** ✓
- Multi-output acquisitions/fitting deferred with clear errors → **Task 3** ✓
- Representation matches gp-ude `ExactGPField` (`d`, `n×d` `α`, `_chol`) → **Task 1** test asserts `size(g.α)==(n,d)` ✓
- `d=1` byte-for-byte unchanged → guarded by the unchanged full suite + the explicit `nlml` d=1 regression test ✓

**Placeholder scan:** none — every step has complete code and exact commands.

**Type consistency:** `_obsmatrix(y, d)` returns `Vector` (d=1) / `Matrix` (d>1), consumed identically by `update`/`_update_incremental` (via `δnew`) and reflected in `size(g.α)`. `_outputdim` (Task 3) returns `Int`, matching the `g.d` field. `_obs_eltype(g::ExactGP)` is instance-based and coexists with the type-based `_obs_eltype(::LaplaceGP)=Bool`.

## Follow-on (separate plans / phases)

- **Multi-output acquisitions** — a defined reduction over per-output means against the shared variance (which output? a scalarization?). Lifts the Task-3 `d>1` guards.
- **Kernel parameterization** (`destructure`-based, [decided](../specs/2026-06-22-kernel-parameterization-design.md)) — generalizes `fit`; would also be where multi-output fitting (if wanted) is addressed.
- **Convergence with gp-ude's `SparseGP <: AbstractGPModel`** — full unification of the two GP families.
