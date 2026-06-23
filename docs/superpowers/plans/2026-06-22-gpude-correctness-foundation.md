# GP-UDE Correctness Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every uncertainty number the GP-UDE bridge emits trustworthy — SVGP variational variance trained by data, PULL covariances numerically valid, observation noise per-dimension — each pinned by a test that fails on regression.

**Architecture:** Pure GP/SVGP math lives in `src/gpude.jl`; solve-touching code (loss, training, propagation) in `ext/MagpieSciMLExt.jl`; metrics in `src/eval.jl`. The SVGP calibration fix threads a per-output field-variance closure (`uvar`) out of each field's `field_rhs` and into the field-agnostic `shooting_data_term`, evaluated post-solve on `Array(sol)` (R1-safe, Mooncake-composable). The per-dimension `σ_obs` change widens the trained-vector hyper-prefix through the single `nhyp(field)` layout chokepoint.

**Tech Stack:** Julia ≥1.10, AbstractGPs/KernelFunctions, OrdinaryDiffEq + SciMLSensitivity (`GaussAdjoint`+`MooncakeVJP`), Mooncake (outer AD via DifferentiationInterface), Optimization (ADAM→LBFGS), StatsFuns. Tests use `Test`, `FiniteDifferences` for gradient gates.

## Global Constraints

- Julia compat floor: `julia = "1.10"` (Project.toml). Code must not use >1.10 syntax.
- All Cholesky factorizations go through the internal `_chol` chokepoint (`cholesky(Symmetric(K); check=false)`). Never call `cholesky` directly.
- **R1:** never closure-capture `α` (or any trained quantity) *into* the ODE `rhs!`/`pf`. Post-solve closures (like `uvar`) capturing variational params are fine.
- **R2:** extract ODE solutions via `Array(sol)`, never `sol[:, i]`.
- Through-solver loss code must stay Mooncake-clean: no `try/catch`, no `@debug`, no convergence branches, no non-smooth `max(0, ·)` clamps on the AD path (use no-clamp or `softplus`).
- Relative jitter convention: in-loss and posterior-reconstruction Cholesky use `jitter·σ²` (`exp(lognoise + 2logσ)` for Exact, `field.jitter·exp(2logσ)` for SVGP). Do not weaken — it is the *only* backward-pass protection.
- Formatting: Runic.jl (`runic --inplace`) before every commit. CI enforces it.
- SciML-dependent tests are gated behind `ENV["MAGPIE_TEST_SCIML"] == "true"`.
- Run a single test file with: `julia --project=. test/<file>.jl` (set the env flag inline when needed).

---

## File Structure

- `src/eval.jl` — **modify**: add `pathwise_moments`; export it.
- `src/gpude.jl` — **modify**: replace `const NHYP=3` with `outputdim`/`nhyp`; fix scalar `σ_obs` reads in `hyp`/`unpack`; widen ctor `v0`; add an AD-safe loss-side `svgp_var` (no clamp).
- `ext/MagpieSciMLExt.jl` — **modify**: `uvar` closure in all three `field_rhs` methods; thread it + per-dim `σ_obs` through `field_loss`/`shooting_data_term`/`build_loss`; `_project_psd` + full-history buffer + aggregated warn in `pull_propagate`; divergence-guard docs.
- `src/Magpie.jl` — **modify**: export `pathwise_moments` (and `nhyp`/`outputdim` stay unexported `Magpie.`-qualified).
- `test/test_eval.jl` — **modify**: `pathwise_moments` test; de-tautologize the "perfect→1.0" coverage test.
- `test/test_gpude_pull.jl` — **modify**: `_project_psd` unit; SVGP PULL coverage oracle.
- `test/test_gpude_noise.jl` — **modify**: tighten `σ_obs` band; add multi-output heterogeneous-noise recovery.
- `test/test_gpude_svgp_mo.jl` — **modify**: extend the MO-ELBO gradient gate to the trace-corrected loss.
- `test/test_gpude_calibration.jl` — **create**: the §2 differential calibration oracle.
- `test/test_gpude_guard.jl` — **create**: divergence-guard trigger test.
- `test/test_gpude_stage2.jl` — **modify**: add `train!`→MultipleShooting (ExactGPField) end-to-end test.
- `test/runtests.jl` — **modify**: include the two new test files under the `MAGPIE_TEST_SCIML` gate.

Task order respects dependencies: helper/isolated fixes first (1–3), then the layout change (4), then per-dim data term (5), then the SVGP trace correction that composes with both (6), then the oracle (7), then remaining test hardening (8).

---

### Task 1: `pathwise_moments` helper

**Files:**
- Modify: `src/eval.jl` (append after `ridge_slice`)
- Modify: `src/Magpie.jl` (export line for eval helpers, ~line 45)
- Test: `test/test_eval.jl`

**Interfaces:**
- Produces: `pathwise_moments(ens::AbstractArray{<:Real,3}) -> (μs::Vector{Vector{Float64}}, Σs::Vector{Matrix{Float64}})`. `ens` is `N×d×T` (samples × dim × timestep); `μs[k]` = length-`d` sample mean at step `k`, `Σs[k]` = `d×d` sample covariance. Consumes nothing from other tasks. Consumed by Tasks 7, 8.

- [ ] **Step 1: Write the failing test**

Add to `test/test_eval.jl` inside the top-level `@testset`:

```julia
@testset "pathwise_moments" begin
    # Deterministic ensemble: 3 samples, d=2, T=2.
    ens = zeros(3, 2, 2)
    ens[:, :, 1] = [1.0 0.0; 3.0 0.0; 5.0 0.0]   # step1: x-col mean 3, var 4; y-col 0
    ens[:, :, 2] = [0.0 2.0; 0.0 4.0; 0.0 6.0]   # step2: x 0; y mean 4, var 4
    μs, Σs = pathwise_moments(ens)
    @test length(μs) == 2 && length(Σs) == 2
    @test μs[1] ≈ [3.0, 0.0]
    @test μs[2] ≈ [0.0, 4.0]
    @test Σs[1][1, 1] ≈ 4.0           # sample var of [1,3,5] = 4
    @test size(Σs[1]) == (2, 2)
    @test issymmetric(Σs[2])
    # round-trips into coverage without error
    truth = [μs[k] for k in 1:2]
    @test coverage(truth, μs, Σs .+ Ref(0.1I); level = 0.9) ≥ 0.0
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_eval.jl`
Expected: FAIL — `UndefVarError: pathwise_moments not defined`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/eval.jl`:

```julia
# ---------------------------------------------------------------------------
# pathwise_moments
# ---------------------------------------------------------------------------

"""
    pathwise_moments(ens) -> (μs, Σs)

Convert a Pathwise ensemble `ens` of shape `N × d × T` (samples × dimension ×
time-step) into per-step posterior moments consumable by [`coverage`](@ref):
`μs[k]` is the length-`d` sample mean and `Σs[k]` the `d×d` sample covariance at
time-step `k`. Samples are the rows of each `ens[:, :, k]` slice.
"""
function pathwise_moments(ens::AbstractArray{<:Real, 3})
    N, d, T = size(ens)
    N ≥ 2 || throw(ArgumentError("pathwise_moments needs ≥2 samples, got N=$N"))
    μs = [vec(mean(@view(ens[:, :, k]); dims = 1)) for k in 1:T]
    Σs = [Matrix(cov(@view(ens[:, :, k]))) for k in 1:T]   # cov over rows → d×d
    return μs, Σs
end
```

Add `pathwise_moments` to the eval export line in `src/Magpie.jl`:

```julia
export coverage, field_error, recovery_metrics, ridge_slice, pathwise_moments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/test_eval.jl`
Expected: PASS (all `test_eval.jl` testsets).

- [ ] **Step 5: Refactor the four examples to use the helper (DRY)**

In each of `examples/gp_ude_lotka_volterra.jl`, `gp_ude_vanderpol.jl`, `gp_ude_scale_forcing.jl`, `gp_ude_fitzhugh_nagumo.jl`, replace the inline pair
```julia
μs_path = [vec(mean(ens[:, :, k]; dims = 1)) for k in 1:nsteps]
Σs_path = [cov(ens[:, :, k]) for k in 1:nsteps]
```
with
```julia
μs_path, Σs_path = pathwise_moments(ens)
```
(match each file's local variable names for the ensemble and step count). Leave all surrounding assertions unchanged.

- [ ] **Step 6: Verify examples still parse**

Run: `julia --project=docs -e 'include("examples/gp_ude_lotka_volterra.jl")'`
Expected: runs to completion, the example's `#src` `@test cov90_path ≥ 0.6` still passes.

- [ ] **Step 7: Format and commit**

```bash
runic --inplace src/eval.jl src/Magpie.jl examples/gp_ude_*.jl
git add src/eval.jl src/Magpie.jl test/test_eval.jl examples/gp_ude_*.jl
git commit -m "feat(eval): pathwise_moments helper; DRY example moment extraction"
```

---

### Task 2: PULL PSD-cone projection + full-history buffer

**Files:**
- Modify: `ext/MagpieSciMLExt.jl` — add `_project_psd`; replace the diagonal clamp in `pull_propagate` (lines ~436-439); change `buffer` default in the two `propagate` methods (~582, ~613) and `pull_propagate` (~426).
- Test: `test/test_gpude_pull.jl`

**Interfaces:**
- Produces: `_project_psd(Σ::AbstractMatrix) -> Matrix` (nearest-PSD with a small relative floor; always positive-definite when the input has any positive eigenvalue). Internal to the ext.

- [ ] **Step 1: Write the failing test**

Add to `test/test_gpude_pull.jl` (this file already runs under `MAGPIE_TEST_SCIML`; access the ext module via the pattern already used there — at the top it does `ext = Base.get_extension(Magpie, :MagpieSciMLExt)`):

```julia
@testset "project_psd" begin
    # Indefinite symmetric matrix: eigenvalues {2, -1}.
    M = [0.5 1.5; 1.5 0.5]
    @test !isposdef(M)
    P = ext._project_psd(M)
    @test isposdef(P)                      # PD after projection
    @test issymmetric(P)
    # nearest-PSD: positive eigenpair preserved, negative clamped to a small floor
    ev = sort(eigen(Symmetric(P)).values)
    @test ev[2] ≈ 2.0 rtol = 1e-6          # the +2 eigenvalue survives
    @test 0 < ev[1] < 1e-6 * ev[2] * 10    # the −1 eigenvalue lifted to ~relative floor
    # already-PSD input is left essentially unchanged
    G = [2.0 0.3; 0.3 1.0]
    @test ext._project_psd(G) ≈ G rtol = 1e-6
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_pull.jl`
Expected: FAIL — `_project_psd` not defined.

- [ ] **Step 3: Implement `_project_psd` and wire it into `pull_propagate`**

Add near the top of the PULL section in `ext/MagpieSciMLExt.jl` (after `field_var`, before `pull_propagate`):

```julia
# Project a symmetric matrix onto the PSD cone with a small RELATIVE floor, so the
# result is a valid (positive-definite) covariance. Forward-only path — never
# differentiated (see eval.jl header) — so `eigen` is unconstrained here.
function _project_psd(Σ::AbstractMatrix)
    S = Symmetric(Matrix(Σ))
    E = eigen(S)
    λmax = maximum(E.values)
    fl = λmax > 0 ? 1.0e-10 * λmax : eps()      # relative floor ⇒ PD output, cond ≤ 1e10
    λ = max.(E.values, fl)
    return Matrix(Symmetric(E.vectors * Diagonal(λ) * E.vectors'))
end
```

Replace the diagonal clamp block in `pull_propagate` (currently):

```julia
        Σ = Matrix(Symmetric(A * Σ * A' + h^2 .* Matrix(V) + h .* (A * Dn + Dn' * A')))
        for k in 1:d
            Σ[k, k] < 0 && (@warn "PULL: negative variance clamped" step = n; Σ[k, k] = eps())
        end
```

with (introduce a counter `nproj` initialized to `0` just before the `for n in ...` loop):

```julia
        Σraw = Matrix(Symmetric(A * Σ * A' + h^2 .* Matrix(V) + h .* (A * Dn + Dn' * A')))
        minev = minimum(eigen(Symmetric(Σraw)).values)
        Σ = _project_psd(Σraw)
        minev < 0 && (nproj += 1)
```

and after the `for n in ...` loop, before `return μs, Σs`, add:

```julia
    nproj > 0 && @warn "PULL: projected $nproj/$(length(ts)-1) step(s) onto the PSD cone (indefinite moment-matched Σ)"
```

- [ ] **Step 4: Change the `buffer` default to full history**

In `pull_propagate` change the signature default `buffer::Int = 20` → `buffer::Int = typemax(Int)`. In the two `propagate` methods (`AbstractVector{<:ExactGP}` and `AbstractVector{<:SparseGP}`) change `buffer = 20` → `buffer = typemax(Int)`. In `_pull_Dn`, add a one-time truncation note: after computing `lo = max(1, npast - buffer + 1)`, leave behavior unchanged (full history when buffer is huge); no warning needed when `buffer ≥ npast`. (Truncation only happens if a user explicitly passes a finite `buffer < npast`; document in the `propagate` docstring that a finite buffer biases Σ downward.)

Update the `propagate` docstrings: change `buffer` description to "number of past cross-covariance terms retained (default: full history); a finite value biases Σ downward."

- [ ] **Step 5: Run tests to verify they pass**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_pull.jl`
Expected: PASS — the new `project_psd` testset AND all existing PULL oracles (brute-force telescope `< 1e-9`, n=2 anchor `< 1e-12`, eq-21b order-of-magnitude, PULL-vs-Pathwise `rel < 0.25`). The buffer-truncation testset (which passes explicit finite buffers) still passes because it sets `buffer` explicitly.

- [ ] **Step 6: Format and commit**

```bash
runic --inplace ext/MagpieSciMLExt.jl
git add ext/MagpieSciMLExt.jl test/test_gpude_pull.jl
git commit -m "fix(gpude): PULL PSD-cone projection + full-history buffer default"
```

---

### Task 3: Divergence-guard honesty (docs + trigger test)

**Files:**
- Modify: `ext/MagpieSciMLExt.jl` — comment at the SingleShooting guard (~line 157-160).
- Create: `test/test_gpude_guard.jl`
- Modify: `test/runtests.jl` — include the new file under the SciML gate.

**Interfaces:** none produced; consumes `Magpie.train!`/`field_loss` (existing).

- [ ] **Step 1: Write the failing test**

Create `test/test_gpude_guard.jl`:

```julia
using Test, Magpie, LinearAlgebra
using OrdinaryDiffEq, SciMLSensitivity
using KernelFunctions, AbstractGPs

ext = Base.get_extension(Magpie, :MagpieSciMLExt)

@testset "divergence guard returns finite sentinel" begin
    # A field whose RHS blows the ODE up over the tspan, so Array(sol) is non-finite.
    Z = [[x] for x in range(-1, 1; length = 4)]
    field = ExactGPField(Magpie._kernel(0.0, 0.0), Z; d = 1)
    # Huge positive weights ⇒ strongly self-amplifying du = +k·α·... ⇒ blow-up.
    v = copy(field.v0)
    v[(Magpie.NHYP + 1):end] .= 1.0e6        # field weights → divergent dynamics
    ts = collect(range(0.0, 5.0; length = 20))
    X = zeros(1, length(ts))                  # data is irrelevant; we only check the sentinel
    loss = ext.field_loss(field, Magpie.SingleShooting(), [(ts, X)]; u0 = [10.0], tspan = (0.0, 5.0))
    L = loss(v)
    @test isfinite(L)                         # guard converts blow-up to a finite sentinel
    @test L ≥ 1.0e6                            # the sentinel value (plus regularizer)
end
```

- [ ] **Step 2: Run test to verify it fails (or errors)**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_guard.jl`
Expected: Either FAIL (if the integrator returns NaN without the guard firing — confirming we need the guard) or PASS if the guard already catches it. If it PASSES, the test now *locks in* the guard behavior (its purpose). If it FAILS because the solve throws before returning, adjust `v` magnitude down (e.g. `1.0e3`) until the integrator saves non-finite points rather than throwing, so the post-solve guard path is exercised.

- [ ] **Step 3: Add the honesty comment**

At the SingleShooting divergence guard in `shooting_data_term` (the `(size(A) == size(X) && all(isfinite, A)) || return convert(T, 1.0e6)` line), expand the existing comment to state:

```julia
        # Divergence guard → finite sentinel (constant ⇒ zero gradient: "don't step here").
        # SCOPE: this guards only the FORWARD solve (NaN/Inf state or wrong shape). It does
        # NOT protect the Mooncake Cholesky-solve BACKWARD (potrs SingularException) — that
        # throws out of DI.gradient and never reaches here. The RELATIVE jitter
        # (exp(lognoise+2logσ) / field.jitter·σ²) is the SOLE backward guard; do not weaken it
        # on the assumption this sentinel covers it.
        (size(A) == size(X) && all(isfinite, A)) || return convert(T, 1.0e6)
```

- [ ] **Step 4: Register the test file**

In `test/runtests.jl`, inside the `MAGPIE_TEST_SCIML` block, add:

```julia
    include("test_gpude_guard.jl")
```

- [ ] **Step 5: Run to verify pass**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_guard.jl`
Expected: PASS.

- [ ] **Step 6: Format and commit**

```bash
runic --inplace ext/MagpieSciMLExt.jl test/test_gpude_guard.jl test/runtests.jl
git add ext/MagpieSciMLExt.jl test/test_gpude_guard.jl test/runtests.jl
git commit -m "test(gpude): divergence-guard trigger test + forward-only scope doc"
```

---

### Task 4: Per-dimension `σ_obs` — layout chokepoint

**Files:**
- Modify: `src/gpude.jl` — `NHYP`→`outputdim`/`nhyp` (~75); `hyp` (~152); `wmat` (~154); `svgp_Z`/`svgp_μ`/`svgp_Lsblk` (~245-254); `unpack(::ExactGPField)` (~163); `unpack(::SVGPField)` (~264); ctors (~132-139, ~223-233).
- Test: `test/test_gpude_unit.jl` (already always-run, not SciML-gated).

**Interfaces:**
- Produces: `Magpie.outputdim(::ExactGPField)=f.d`, `Magpie.outputdim(::SVGPField)=f.dout`; `Magpie.nhyp(field)=2+outputdim(field)`. Hyper-prefix of the trained vector becomes `[logℓ, logσ, logσ_obs(1..d)]`. `unpack(field, v).logσ_obs` is now a length-`d` `Vector`. Consumed by Task 5 (data term) and Task 6 (trace divisor).

- [ ] **Step 1: Write the failing test**

Add to `test/test_gpude_unit.jl`:

```julia
@testset "per-dim σ_obs layout" begin
    # ExactGPField, d=2: v0 prefix is [logℓ, logσ, logσ_obs(1), logσ_obs(2), w...]
    Z = [[x] for x in range(-1, 1; length = 3)]
    f = ExactGPField(Magpie._kernel(0.0, 0.0), Z; d = 2)
    @test Magpie.outputdim(f) == 2
    @test Magpie.nhyp(f) == 4
    @test length(f.v0) == 4 + 3 * 2            # 2 hypers + 2 σ_obs + n*d weights
    up = Magpie.unpack(f, f.v0)
    @test up.logσ_obs isa AbstractVector
    @test length(up.logσ_obs) == 2
    @test up.logσ_obs ≈ fill(log(0.1), 2)
    @test size(up.w) == (3, 2)                 # w-block read correctly after the wider prefix
    # SVGPField, dout=2: prefix [logℓ, logσ, logσ_obs(1), logσ_obs(2), Z..., μ..., L_S...]
    sf = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 2)
    @test Magpie.outputdim(sf) == 2
    @test Magpie.nhyp(sf) == 4
    ups = Magpie.unpack(sf, sf.v0)
    @test length(ups.logσ_obs) == 2
    @test size(ups.Z) == (1, 3)                # D×M read correctly after wider prefix
    @test size(ups.μ) == (3, 2)                # M×dout
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_gpude_unit.jl`
Expected: FAIL — `outputdim`/`nhyp` not defined; `length(f.v0)` wrong (still has 1 σ_obs slot).

- [ ] **Step 3: Replace the layout chokepoint**

In `src/gpude.jl`, replace `const NHYP = 3` (and its comment block) with:

```julia
# SINGLE SOURCE OF TRUTH for the hyper-prefix width. The trained vector ALWAYS begins
# `[logℓ, logσ, logσ_obs(1..d)]`, where the d-vector logσ_obs (the observation-noise
# log-stds, ONE per output dimension) is used ONLY by the data term (Gaussian NLL) and is
# NOT threaded into the solve param `pf`. Every field-specific block offset is `nhyp(field) + …`,
# so changing the hyper layout touches only these accessors. (The in-loss SOLVE param `pf` —
# built by each field's `field_rhs` — carries its own separate layout and never holds σ_obs.)
outputdim(f::ExactGPField) = f.d
outputdim(f::SVGPField) = f.dout
nhyp(f) = 2 + outputdim(f)                      # [logℓ, logσ, logσ_obs(1..d)]
```

(`CompositeField` resolves `outputdim` via its `getproperty` forward of `d`/`dout` to the inner field — no extra method needed; confirm by a quick `outputdim(cf)` works because `cf.d`/`cf.dout` forward.)

Add `outputdim(cf::CompositeField) = outputdim(cf.gp)` explicitly to avoid relying on the forward for a function (functions don't dispatch through `getproperty`):

```julia
outputdim(cf::CompositeField) = outputdim(cf.gp)
```

Place these three `outputdim` methods after the `CompositeField`/`ExactGPField`/`SVGPField` struct definitions (all are defined before `const NHYP` today; keep `outputdim` after the last of them, and `nhyp` after `outputdim`).

- [ ] **Step 4: Fix the layout accessors and the two scalar `v[3]` reads**

`hyp` (FieldLayout-based, used by ExactGPField) — change to read the σ_obs **block** (FieldLayout carries `L.d`):

```julia
# hyp exposes ALL hyper slots; logσ_obs is a d-vector consumed only by the data term.
hyp(L::FieldLayout, v) = (logℓ = v[1], logσ = v[2], logσ_obs = v[3:(2 + L.d)])
```

`wmat` — w-block now starts after the wider prefix:

```julia
wmat(L::FieldLayout, v) = reshape(v[(2 + L.d + 1):(2 + L.d + nw(L))], L.n, L.d)
```

(`nw(L) = L.n * L.d` unchanged.)

SVGP accessors — replace `NHYP` with `2 + f.dout`:

```julia
svgp_Z(f::SVGPField, v) = reshape(v[(2 + f.dout + 1):(2 + f.dout + f.D * f.M)], f.D, f.M)

svgp_μ(f::SVGPField, v) = reshape(
    v[(2 + f.dout + f.D * f.M + 1):(2 + f.dout + f.D * f.M + f.M * f.dout)], f.M, f.dout
)

function svgp_Lsblk(f::SVGPField, v, i)
    o = 2 + f.dout + f.D * f.M + f.M * f.dout
    return v[(o + (i - 1) * nLS(f.M) + 1):(o + i * nLS(f.M))]
end
```

`unpack(::SVGPField)` — scalar `v[3]` → block:

```julia
function unpack(field::SVGPField, v)
    return (
        logℓ = v[1], logσ = v[2], logσ_obs = v[3:(2 + field.dout)], Z = svgp_Z(field, v),
        μ = svgp_μ(field, v),
        Ls = [unpack_LS(svgp_Lsblk(field, v, i), field.M) for i in 1:field.dout],
    )
end
```

`unpack(::ExactGPField)` already forwards `h.logσ_obs` from `hyp` — no change beyond `hyp` (its `logσ_obs` field is now a vector).

- [ ] **Step 5: Widen the constructor `v0` blocks**

`ExactGPField` constructor:

```julia
function ExactGPField(
        kernel::Kernel, Z; d::Int = 1, mean = AbstractGPs.ZeroMean(),
        logℓ0 = 0.0, logσ0 = 0.0, logσ_obs0 = log(0.1), lognoise = log(1.0e-2)
    )
    n = length(Z)
    σobs0 = logσ_obs0 isa AbstractVector ? collect(float.(logσ_obs0)) : fill(float(logσ_obs0), d)
    @assert length(σobs0) == d "logσ_obs0 must be a scalar or length-d vector"
    v0 = vcat(logℓ0, logσ0, σobs0, zeros(n * d))
    return ExactGPField(AbstractGPs.GP(mean, kernel), collect(Z), n, d, Float64(lognoise), v0)
end
```

`SVGPField` constructor:

```julia
function SVGPField(
        kernel::Kernel, Z0::AbstractVector; dout::Int = 1, mean = AbstractGPs.ZeroMean(),
        logℓ0 = 0.0, logσ0 = 0.0, logσ_obs0 = log(0.1), jitter = 1.0e-4
    )
    M = length(Z0); D = length(first(Z0))
    σobs0 = logσ_obs0 isa AbstractVector ? collect(float.(logσ_obs0)) : fill(float(logσ_obs0), dout)
    @assert length(σobs0) == dout "logσ_obs0 must be a scalar or length-dout vector"
    μ0 = zeros(M * dout)
    Ls0 = reduce(vcat, [vcat(zeros(M), zeros(nLS(M) - M)) for _ in 1:dout])
    v0 = vcat(logℓ0, logσ0, σobs0, reduce(vcat, Z0), μ0, Ls0)
    return SVGPField(AbstractGPs.GP(mean, kernel), collect(Z0), M, dout, D, Float64(jitter), v0)
end
```

- [ ] **Step 6: Run the layout test**

Run: `julia --project=. test/test_gpude_unit.jl`
Expected: PASS (new layout testset + existing unit tests — the SVGP-math tests in this file must still pass, confirming the accessors read correctly).

- [ ] **Step 7: Commit (data-term wiring follows in Task 5 — the suite is not green yet for SciML paths)**

> Note: after this task the *non-SciML* unit tests pass, but the ext data-term code in `ext/MagpieSciMLExt.jl` still reads `v[Magpie.NHYP]` (now undefined) — that is fixed in Task 5. Commit the layout change on its own so the diff stays reviewable; do NOT run the full SciML suite between Task 4 and Task 5.

```bash
runic --inplace src/gpude.jl test/test_gpude_unit.jl
git add src/gpude.jl test/test_gpude_unit.jl
git commit -m "refactor(gpude): per-dim σ_obs layout (NHYP→nhyp(field)); fix scalar v[3] reads"
```

---

### Task 5: Per-dimension `σ_obs` — data term

**Files:**
- Modify: `ext/MagpieSciMLExt.jl` — `_gaussian_nll` (~132); SingleShooting `shooting_data_term` (~143-165); MultipleShooting `shooting_data_term` (~176-201); `field_loss` (~44-49); MS `build_loss` (~225-241).
- Test: `test/test_gpude_noise.jl`

**Interfaces:**
- Consumes: `Magpie.outputdim`, `Magpie.nhyp`, `unpack(...).logσ_obs::Vector` (Task 4).
- Produces: data term computes a per-output Gaussian NLL `Σ_j [SSE_j/(2σ²_obs,j) + (N_j/2)log(2π σ²_obs,j)]`. `shooting_data_term` now accepts a `uvar` kwarg (defaulted no-op here; populated in Task 6).

- [ ] **Step 1: Write the failing test (heterogeneous-noise recovery)**

Add to `test/test_gpude_noise.jl`:

```julia
@testset "per-dim σ_obs recovers heterogeneous noise" begin
    rng = MersenneTwister(11)
    # 2-output linear field du = A u, with very different per-dim observation noise.
    Atrue = [-0.3 0.0; 0.0 -0.5]
    truef(u, t) = Atrue * u
    u0 = [1.0, 1.0]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 40))
    sol = solve(ODEProblem((du, u, p, t) -> (du .= Atrue * u), u0, tspan), Tsit5(); saveat = ts)
    Xclean = Array(sol)
    σ1, σ2 = 0.02, 0.20                                   # 10× noise asymmetry
    X = copy(Xclean); X[1, :] .+= σ1 .* randn(rng, length(ts)); X[2, :] .+= σ2 .* randn(rng, length(ts))
    Z = [collect(c) for c in eachcol(Xclean[:, 1:8])]
    field = ExactGPField(Magpie._kernel(0.0, 0.0), Z; d = 2)
    _, vfit = train!(field, (ts, X); shooting = SingleShooting(), adam_iters = 400, maxiters = 100)
    σobs = exp.(Magpie.unpack(field, field.v0).logσ_obs)
    # Recovered per-dim σ_obs ordering matches the true asymmetry, each within a tight band.
    @test σobs[2] > 2 * σobs[1]                            # dim-2 noisier, clearly separated
    @test 0.7 * σ1 < σobs[1] < 1.6 * σ1
    @test 0.7 * σ2 < σobs[2] < 1.6 * σ2
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_noise.jl`
Expected: FAIL/ERROR — `field_loss` references `v[Magpie.NHYP]` (undefined after Task 4); even once that is fixed, a single pooled σ_obs cannot separate the two scales.

- [ ] **Step 3: Vectorize `_gaussian_nll` callers (per-output, summed scalar form)**

Keep `_gaussian_nll` scalar (per output), and sum over outputs at the call sites (this avoids the vector-broadcast landmine). `_gaussian_nll` is unchanged:

```julia
_gaussian_nll(sse, Nd, logσ_obs) = (σ2 = exp(2 * logσ_obs); sse / (2σ2) + (Nd / 2) * log(2π * σ2))
```

Rewrite the SingleShooting `shooting_data_term` accumulation. Signature gains `uvar` (defaulted no-op):

```julia
function shooting_data_term(
        field, ::Magpie.SingleShooting, (pf, rhs!), data;
        logσ_obs, u0 = nothing, tspan = nothing, known_physics = (u, t) -> zero(u),
        uvar = nothing, solver = Tsit5(), sensealg = DEFAULT_SENSEALG, _kw...
    )
    f!(du, u, p, t) = rhs!(du, u, p, t; known_physics)
    T = eltype(pf)
    dout = length(logσ_obs)
    sse = zeros(T, dout)
    trace = zeros(T, dout)
    Nd = zeros(Int, dout)
    for (ts, X) in data
        ic = u0 === nothing ? collect(X[:, 1]) : u0
        tsp = tspan === nothing ? (first(ts), last(ts)) : tspan
        sol = solve(ODEProblem(f!, ic, tsp, pf), solver; saveat = ts, sensealg)
        A = Array(sol)                                  # R2
        (size(A) == size(X) && all(isfinite, A)) || return convert(T, 1.0e6)
        sse .+= vec(sum(abs2, A .- X; dims = 2))        # per-output SSE
        Nd .+= size(X, 2)                                # per-output count = #timepoints
        if uvar !== nothing
            for col in eachcol(A)
                trace .+= uvar(col)                      # per-output field variance (Task 6)
            end
        end
    end
    nll = sum(_gaussian_nll(sse[j], Nd[j], logσ_obs[j]) for j in 1:dout)
    tracecorr = sum(trace[j] / (2 * exp(2 * logσ_obs[j])) for j in 1:dout)
    return nll + tracecorr
end
```

Rewrite the MultipleShooting `shooting_data_term` data-misfit per output (penalties unchanged):

```julia
function shooting_data_term(
        field, ms::Magpie.MultipleShooting, (pf, rhs!), data;
        logσ_obs, s0, known_physics = (u, t) -> zero(u),
        uvar = nothing, solver = Tsit5(), sensealg = DEFAULT_SENSEALG, _kw...
    )
    (ts, X) = only(data)
    S = ms.nsegments
    seg_idx = round.(Int, range(1, length(ts); length = S + 1))
    seg_t = [ts[i] for i in seg_idx]
    f!(du, u, p, t) = rhs!(du, u, p, t; known_physics)
    dout = length(logσ_obs)
    dataerr = zeros(eltype(pf), dout)
    cont = zero(eltype(pf))
    Nd = zeros(Int, dout)
    for i in 1:S
        sol = solve(
            ODEProblem(f!, s0[:, i], (seg_t[i], seg_t[i + 1]), pf), solver;
            saveat = [seg_t[i + 1]], sensealg
        )
        endp = Array(sol)[:, end]
        all(isfinite, endp) || return convert(eltype(pf), 1.0e6)
        dataerr .+= abs2.(endp .- X[:, seg_idx[i + 1]])
        Nd .+= 1
        i < S && (cont += sum(abs2, endp .- s0[:, i + 1]))
    end
    nll = sum(_gaussian_nll(dataerr[j], Nd[j], logσ_obs[j]) for j in 1:dout)
    return nll + ms.λ * cont + ms.λ0 * sum(abs2, s0[:, 1] .- X[:, 1])
end
```

- [ ] **Step 4: Thread the `σ_obs` block through `field_loss` and `build_loss`**

`field_loss` — extract the σ_obs **block** and pass it (plus the future `uvar`; in Task 5 `field_rhs` still returns a 2-tuple, so destructure only `(pf, rhs!)` and pass no `uvar` yet — Task 6 adds it):

```julia
function field_loss(field, shooting, data; kw...)
    return function (v)
        pf, rhs! = field_rhs(field, v)
        return shooting_data_term(
            field, shooting, (pf, rhs!), data;
            logσ_obs = v[3:(2 + Magpie.outputdim(field))], kw...
        ) + Magpie.regularizer(field, v; kw...)
    end
end
```

MS `build_loss` — fix the s0 offset and the σ_obs block:

```julia
function build_loss(
        field::ExactGPField, L::FieldLayout, u_data, t_data, tspan,
        ms::Magpie.MultipleShooting; kw...
    )
    S = ms.nsegments
    nwL = L.n * L.d
    off = 2 + field.d + nwL                                  # after [logℓ,logσ,logσ_obs(1..d),vec(w)]
    regkw = merge((λσ = 0.0,), values(kw))
    return function loss(v)
        s0 = reshape(v[(off + 1):(off + field.d * S)], field.d, S)
        return shooting_data_term(
            field, ms, field_rhs(field, v), [(collect(t_data), u_data)];
            s0 = s0, logσ_obs = v[3:(2 + field.d)], kw...
        ) + Magpie.regularizer(field, v; regkw...)
    end
end
```

(Also update `_init_vec(::MultipleShooting)` only if it indexed the prefix — it does not; it appends `vec(s0)` after `field.v0`, which is already the wider vector. No change.)

- [ ] **Step 5: Run the recovery test**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_noise.jl`
Expected: PASS — including the existing exact/SVGP σ_obs recovery testsets (now reading `logσ_obs[1]` as a 1-vector for d=1 fields — update those existing assertions from `σ_obs = exp(only(...logσ_obs...))` to `exp(Magpie.unpack(field, field.v0).logσ_obs[1])` where they currently assume a scalar).

- [ ] **Step 6: Run the broader SciML suite to confirm no layout regression**

Run: `MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS (134+ tests). This is the first full-suite checkpoint after the layout change.

- [ ] **Step 7: Format and commit**

```bash
runic --inplace ext/MagpieSciMLExt.jl test/test_gpude_noise.jl
git add ext/MagpieSciMLExt.jl test/test_gpude_noise.jl
git commit -m "feat(gpude): per-dimension σ_obs data term (heterogeneous observation noise)"
```

---

### Task 6: SVGP calibrated ELBO — `uvar` trace correction

**Files:**
- Modify: `src/gpude.jl` — add AD-safe `svgp_var` (no clamp).
- Modify: `ext/MagpieSciMLExt.jl` — `uvar` in all three `field_rhs` methods; destructure the 3-tuple in `field_loss` and pass `uvar`; update `CompositeField`/MS `field_rhs` call sites.
- Test: `test/test_gpude_svgp_mo.jl` (gradient gate), `test/test_gpude_svgp.jl` (unit).

**Interfaces:**
- Consumes: Task 4 layout, Task 5 `shooting_data_term`'s `uvar` kwarg + `trace` accumulation.
- Produces: `field_rhs(field, v) -> (pf, rhs!, uvar)` for all fields. `uvar(u::AbstractVector) -> Vector` (length = output dim) gives per-output field variance. `Magpie.svgp_var(prior, Z, L_ZZ, L_S, u) -> Real` (AD-safe, no clamp).

- [ ] **Step 1: Write the failing unit test (uvar == svgp_moments variance, minus clamp)**

Add to `test/test_gpude_svgp.jl`:

```julia
@testset "svgp_var is AD-safe svgp_moments variance" begin
    Z = [[x] for x in range(-1, 1; length = 4)]
    k = Magpie._kernel(0.0, 0.0)
    prior = AbstractGPs.GP(AbstractGPs.ZeroMean(), k)
    L_ZZ = Magpie.L_ZZ_factor(prior, Z; jitter = 1e-4)
    L_S = Magpie.unpack_LS([0.1, 0.0, 0.2, 0.0, 0.0, 0.3, -0.1, 0.0, 0.05, 0.15][1:Magpie.nLS(4)], 4)
    u = [0.3]
    α = L_ZZ' \ zeros(4)
    μ_ref, σ2_ref = Magpie.svgp_moments(prior, Z, L_ZZ, α, L_S, u)
    σ2 = Magpie.svgp_var(prior, Z, L_ZZ, L_S, u)
    @test σ2 ≈ σ2_ref rtol = 1e-10           # identical where the clamp is inactive (σ²>0)
    @test σ2 > 0
end
```

- [ ] **Step 2: Run to verify it fails**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_svgp.jl`
Expected: FAIL — `svgp_var` not defined.

- [ ] **Step 3: Add the AD-safe `svgp_var` (no clamp)**

In `src/gpude.jl`, after `svgp_moments`, add:

```julia
"""
    svgp_var(prior, Z, L_ZZ, L_S, u) -> Real

Whitened SVGP predictive variance at a single point `u`, WITHOUT the `max(0,·)` clamp in
[`svgp_moments`](@ref). This is the through-solver-AD-safe form used by the ELBO trace
correction (the clamp's subgradient kink would kill the `L_S` gradient where it matters).
The relative jitter keeps `K_ZZ` conditioned so the unclamped value stays positive in practice.

    A = L_ZZ \\ k(Z,u);   σ² = k(u,u) − A'A + ‖L_S'A‖²
"""
function svgp_var(prior, Z, L_ZZ, L_S, u)
    kZu = vec(AbstractGPs.cov(prior, Z, [u]))
    A = L_ZZ \ kZu
    return only(AbstractGPs.var(prior, [u])) - dot(A, A) + sum(abs2, L_S' * A)
end
```

Export it as unexported `Magpie.svgp_var` (no change to `Magpie.jl` export list — it stays `Magpie.`-qualified, used by the ext).

- [ ] **Step 4: Run the unit test**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_svgp.jl`
Expected: PASS.

- [ ] **Step 5: Add `uvar` to all three `field_rhs` methods**

`field_rhs(::ExactGPField, v)` — return a no-op `uvar` as the 3rd value:

```julia
function field_rhs(field::ExactGPField, v)
    L = FieldLayout(field.n, field.d)
    h = Magpie.hyp(L, v)
    α = solve_alpha(field, h.logℓ, h.logσ, field.lognoise, Magpie.wmat(L, v))
    pf = vcat(h.logℓ, h.logσ, vec(α))
    rhs!(du, u, _pf, t; known_physics) = (du .= known_physics(u, t); du .+= gpfield(field, u, _pf); nothing)
    uvar = _ -> zeros(field.d)                       # exact field has no variational-variance term
    return (pf, rhs!, uvar)
end
```

`field_rhs(::SVGPField, v)` — build the per-output variance closure capturing `L_ZZ`, `Zvec`, `k`, `Ls`:

```julia
function field_rhs(field::SVGPField, v)
    M, dout, D = field.M, field.dout, field.D
    logℓ, logσ = v[1], v[2]
    k = Magpie._kernel(logℓ, logσ)
    Z = Magpie.svgp_Z(field, v)        # D×M
    μ = Magpie.svgp_μ(field, v)        # M×dout
    Zvec = [Z[:, j] for j in 1:M]
    jit = field.jitter * exp(2 * logσ)
    L_ZZ = _chol(kernelmatrix(k, Zvec) + jit * I).L
    α = L_ZZ' \ μ
    pf = vcat(logℓ, logσ, vec(Z), vec(α))
    prior = AbstractGPs.GP(field.prior.mean, k)
    Ls = [Magpie.unpack_LS(Magpie.svgp_Lsblk(field, v, i), M) for i in 1:dout]
    uvar = u -> [Magpie.svgp_var(prior, Zvec, L_ZZ, Ls[i], u) for i in 1:dout]
    function rhs!(du, u, _pf, t; known_physics)
        du .= known_physics(u, t)
        kk = Magpie._kernel(_pf[1], _pf[2])
        Zr = reshape(_pf[3:(2 + D * M)], D, M)
        αr = reshape(_pf[(3 + D * M):(2 + D * M + M * dout)], M, dout)
        for i in 1:dout
            du[i] += sum(kk(u, @view Zr[:, j]) * αr[j, i] for j in 1:M)
        end
        return nothing
    end
    return (pf, rhs!, uvar)
end
```

`field_rhs(::CompositeField, v)` — delegate `uvar` to the inner field; keep the known-physics wrapper:

```julia
function field_rhs(cf::Magpie.CompositeField, v)
    pf, _, uvar = field_rhs(cf.gp, v)        # inner field: pf + its variance closure
    known = cf.known
    function rhs!(du, u, _pf, t; _kw...)
        du .= known(u, t)
        du .+= gpfield(cf.gp, u, _pf)
        return nothing
    end
    return (pf, rhs!, uvar)
end
```

- [ ] **Step 6: Destructure the 3-tuple and pass `uvar` in `field_loss`**

Update `field_loss` (from Task 5) to destructure three values and forward `uvar`:

```julia
function field_loss(field, shooting, data; kw...)
    return function (v)
        pf, rhs!, uvar = field_rhs(field, v)
        return shooting_data_term(
            field, shooting, (pf, rhs!), data;
            logσ_obs = v[3:(2 + Magpie.outputdim(field))], uvar = uvar, kw...
        ) + Magpie.regularizer(field, v; kw...)
    end
end
```

Update the MS `build_loss` call (Task 5) — it calls `field_rhs(field, v)` and passes it to `shooting_data_term`; change to capture and pass `uvar`:

```julia
        pf, rhs!, uvar = field_rhs(field, v)
        return shooting_data_term(
            field, ms, (pf, rhs!), [(collect(t_data), u_data)];
            s0 = s0, logσ_obs = v[3:(2 + field.d)], uvar = uvar, kw...
        ) + Magpie.regularizer(field, v; regkw...)
```

(For ExactGPField MS, `uvar` is the no-op `_ -> zeros(d)`, so the MS trace term is 0 — behavior unchanged.)

- [ ] **Step 7: Write the gradient-gate test (trace-corrected loss is Mooncake-clean)**

Add to `test/test_gpude_svgp_mo.jl` a check that the trace-corrected ELBO loss differentiates and that the trace term actually contributes an `L_S` gradient. Extend the existing FD-vs-Mooncake gate to the full `svgp_elbo_loss` (which now includes the trace term) and assert the gradient wrt an `L_S` slot is non-trivial:

```julia
@testset "trace-corrected SVGP ELBO: gradient sound + L_S coupled" begin
    rng = MersenneTwister(7)
    Z = [[x] for x in range(-1, 1; length = 3)]
    field = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 2)
    ts = collect(range(0.0, 2.0; length = 12))
    Xtrue = hcat([[cos(t), sin(t)] for t in ts]...)
    loss = ext.svgp_elbo_loss(field, [(ts, Xtrue)]; tspan = (0.0, 2.0))
    v = copy(field.v0)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v)[1]
    relerr = norm(g_mc .- g_fd) / (norm(g_fd) + 1e-8)
    @test relerr < 1.0e-3
    # An L_S diagonal slot now receives gradient from the data (was ~0 from KL-only at S=I).
    ls_idx = 2 + field.dout + field.D * field.M + field.M * field.dout + 1   # first L_S raw entry
    @test abs(g_mc[ls_idx]) > 1.0e-6
end
```

- [ ] **Step 8: Run the gradient + unit tests**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_svgp_mo.jl test/test_gpude_svgp.jl`
Expected: PASS — relerr `< 1e-3`, the `L_S` slot gradient is non-trivial, and existing SVGP-math tests still pass.

- [ ] **Step 9: Format and commit**

```bash
runic --inplace src/gpude.jl ext/MagpieSciMLExt.jl test/test_gpude_svgp.jl test/test_gpude_svgp_mo.jl
git add src/gpude.jl ext/MagpieSciMLExt.jl test/test_gpude_svgp.jl test/test_gpude_svgp_mo.jl
git commit -m "feat(gpude): SVGP calibrated ELBO — field-space trace correction trains L_S"
```

---

### Task 7: SVGP differential calibration oracle

**Files:**
- Create: `test/test_gpude_calibration.jl`
- Modify: `test/runtests.jl` — include under the SciML gate.

**Interfaces:**
- Consumes: `train!`, `propagate`, `pathwise_moments` (Task 1), `coverage`, the trace-corrected loss (Task 6). Uses the `regularizer_only` path by constructing the loss WITHOUT the trace term (pass `uvar = nothing` is internal; instead train a second field with the trace disabled — see Step 1's mechanism).

- [ ] **Step 1: Write the calibration oracle**

The differential needs a "trace off" baseline. Mechanism: add an internal kwarg to `train!`/`svgp_elbo_loss` to disable the trace term — `trace = true` default; when `false`, `field_loss` passes `uvar = (_ -> zeros(outputdim))`. Implement that toggle first (small change in `field_loss`):

```julia
function field_loss(field, shooting, data; trace::Bool = true, kw...)
    dout = Magpie.outputdim(field)
    return function (v)
        pf, rhs!, uvar = field_rhs(field, v)
        uv = trace ? uvar : (_ -> zeros(dout))
        return shooting_data_term(
            field, shooting, (pf, rhs!), data;
            logσ_obs = v[3:(2 + dout)], uvar = uv, kw...
        ) + Magpie.regularizer(field, v; kw...)
    end
end
```

(`svgp_elbo_loss` forwards `kw...`, so `trace=false` flows through. `train!` also forwards `kw...` to `_train_loss` → `svgp_elbo_loss`.)

Create `test/test_gpude_calibration.jl`:

```julia
using Test, Magpie, LinearAlgebra, Statistics, Random
using OrdinaryDiffEq, SciMLSensitivity
using KernelFunctions, AbstractGPs

ext = Base.get_extension(Magpie, :MagpieSciMLExt)

@testset "SVGP calibration: trace correction → calibrated + sharp" begin
    rng = MersenneTwister(2026)
    # Linear 1-D field du = a u (a<0): decaying trajectory; SVGP learns the field.
    a = -0.4
    truef(u, t) = [a * u[1]]
    u0 = [1.5]; tspan = (0.0, 5.0); ts = collect(range(tspan...; length = 30))
    Xclean = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = a * u[1]), u0, tspan), Tsit5(); saveat = ts))
    σobs = 0.05
    X = Xclean .+ σobs .* randn(rng, size(Xclean))
    Z = [collect(c) for c in eachcol(Xclean[:, 1:6])]

    # Train two SVGP fields from the SAME init: trace ON vs OFF.
    field_on = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 1)
    field_off = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 1)
    train!(field_on, (ts, X); adam_iters = 600, maxiters = 150, trace = true)
    train!(field_off, (ts, X); adam_iters = 600, maxiters = 150, trace = false)

    # Held-out propagation from a fresh IC; Pathwise ensemble → per-step moments → coverage.
    u0h = [1.2]
    tsh = collect(range(0.0, 5.0; length = 25))
    truth = [collect(c) for c in eachcol(Array(solve(
        ODEProblem((du, u, p, t) -> (du[1] = a * u[1]), u0h, (0.0, 5.0)), Tsit5(); saveat = tsh)))]

    ens_on = propagate(field_on, u0h, (0.0, 5.0); method = Pathwise(256), ts = tsh)
    ens_off = propagate(field_off, u0h, (0.0, 5.0); method = Pathwise(256), ts = tsh)
    μon, Σon = pathwise_moments(ens_on)
    μoff, Σoff = pathwise_moments(ens_off)

    cov_on = coverage(truth, μon, Σon; level = 0.9)
    cov_off = coverage(truth, μoff, Σoff; level = 0.9)
    width_on = mean(only(Σ) for Σ in Σon[2:end])     # mean predictive variance (skip t=0)
    width_off = mean(only(Σ) for Σ in Σoff[2:end])

    @info "SVGP calibration" cov_on cov_off width_on width_off
    # Trace-OFF over-covers with vacuously wide intervals; trace-ON is calibrated AND sharper.
    @test cov_off > 0.97                              # regularized-only ≈ prior variance ⇒ over-covers
    @test abs(cov_on - 0.9) < 0.15                    # trace-corrected ≈ nominal
    @test width_on < 0.6 * width_off                  # and materially sharper (the real property)
end
```

- [ ] **Step 2: Run to verify it fails before Task 6 / passes after**

Run: `MAGPIE_TEST_SCIML=true julia --project=. test/test_gpude_calibration.jl`
Expected: PASS with Task 6 in place. If `cov_on`/`width_on` thresholds are mis-tuned (the spec flags these as empirical), adjust the bands using the `@info` output — but keep the THREE-way conjunction (over-cover off, near-nominal on, sharper on); never loosen to a single absolute band.

- [ ] **Step 3: Register the test**

In `test/runtests.jl`, under the `MAGPIE_TEST_SCIML` block:

```julia
    include("test_gpude_calibration.jl")
```

- [ ] **Step 4: Format and commit**

```bash
runic --inplace ext/MagpieSciMLExt.jl test/test_gpude_calibration.jl test/runtests.jl
git add ext/MagpieSciMLExt.jl test/test_gpude_calibration.jl test/runtests.jl
git commit -m "test(gpude): differential SVGP calibration oracle (calibrated + sharp)"
```

---

### Task 8: Remaining test hardening

**Files:**
- Modify: `test/test_gpude_stage2.jl` — `train!`→MultipleShooting (ExactGPField) e2e.
- Modify: `test/test_eval.jl` — de-tautologize the "perfect→1.0" coverage test.
- Modify: `test/test_gpude_pull.jl` — SVGP/SparseGP PULL coverage oracle.

**Interfaces:** consumes `train!`, `propagate`, `pathwise_moments`, `coverage`.

- [ ] **Step 1: `train!`→MultipleShooting end-to-end (ExactGPField)**

Add to `test/test_gpude_stage2.jl`:

```julia
@testset "train! with MultipleShooting (ExactGPField) runs end-to-end" begin
    rng = MersenneTwister(3)
    a = -0.3
    u0 = [2.0]; tspan = (0.0, 6.0); ts = collect(range(tspan...; length = 36))
    X = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = a * u[1]), u0, tspan), Tsit5(); saveat = ts))
    X .+= 0.02 .* randn(rng, size(X))
    Z = [collect(c) for c in eachcol(X[:, 1:6])]
    field = ExactGPField(Magpie._kernel(0.0, 0.0), Z; d = 1)
    _, vfit = train!(field, (ts, X); shooting = MultipleShooting(nsegments = 4), adam_iters = 300, maxiters = 80)
    @test all(isfinite, vfit)
    gps = posterior_gps(field)
    μs, _ = propagate(gps, u0, tspan; method = PULL(), ts = ts)
    rmse = sqrt(mean(sum(abs2, μs[k] .- X[:, k]) for k in 1:length(ts)))
    @test rmse < 0.5                                  # MS training recovers the trajectory
end
```

- [ ] **Step 2: De-tautologize the coverage "perfect" test**

In `test/test_eval.jl`, replace the `μs = copy(truth)` (exact) test with a level-dependent one that fails if `coverage` ignores `level`/`Σ`:

```julia
@testset "coverage tracks level and scale (not tautological)" begin
    rng = MersenneTwister(5)
    d, n = 2, 4000
    truth = [randn(rng, d) for _ in 1:n]
    μs = [zeros(d) for _ in 1:n]
    Σs = [Matrix{Float64}(I, d, d) for _ in 1:n]      # truth ~ N(0,I), μ=0, Σ=I
    c90 = coverage(truth, μs, Σs; level = 0.9)
    c50 = coverage(truth, μs, Σs; level = 0.5)
    @test abs(c90 - 0.9) < 0.03
    @test abs(c50 - 0.5) < 0.03
    @test c90 > c50                                   # higher level ⇒ more coverage (was ignored by the old test)
    # Under-dispersed Σ ⇒ severe under-coverage (scale matters).
    Σtight = [1e-3 * Matrix{Float64}(I, d, d) for _ in 1:n]
    @test coverage(truth, μs, Σtight; level = 0.9) < 0.2
end
```

- [ ] **Step 3: SVGP/SparseGP PULL coverage oracle**

In `test/test_gpude_pull.jl`, replace the SVGP/SparseGP finiteness smoke testset with a coverage assertion (build a SparseGP field, propagate via PULL, check coverage near nominal on the training trajectory as a sanity oracle):

```julia
@testset "SparseGP PULL coverage is sane" begin
    rng = MersenneTwister(9)
    a = -0.35
    u0 = [1.0]; tspan = (0.0, 5.0); ts = collect(range(tspan...; length = 25))
    X = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = a * u[1]), u0, tspan), Tsit5(); saveat = ts))
    X .+= 0.04 .* randn(rng, size(X))
    Z = [collect(c) for c in eachcol(X[:, 1:6])]
    field = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 1)
    train!(field, (ts, X); adam_iters = 500, maxiters = 120, trace = true)
    truth = [collect(c) for c in eachcol(X)]
    μs, Σs = propagate(field, u0, tspan; method = PULL(), ts = ts)
    # Every Σ is PSD-valid (Task 2/§3) and coverage is finite + not absurd.
    @test all(isposdef, Σs[2:end])
    cov90 = coverage(truth, μs, Σs; level = 0.9)
    @test 0.5 ≤ cov90 ≤ 1.0                            # quantitative oracle, not just finiteness
end
```

- [ ] **Step 4: Run the full suite**

Run: `MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — all tasks integrated; the suite is green end-to-end.

- [ ] **Step 5: Format and commit**

```bash
runic --inplace test/test_gpude_stage2.jl test/test_eval.jl test/test_gpude_pull.jl
git add test/test_gpude_stage2.jl test/test_eval.jl test/test_gpude_pull.jl
git commit -m "test(gpude): train!→MS e2e, de-tautologized coverage, SparseGP PULL oracle"
```

---

## Self-Review

**Spec coverage:**
- §1 SVGP calibrated ELBO → Task 6 (uvar trace correction) + Task 6 gradient gate. ✓
- §2 calibration oracle + `pathwise_moments` → Task 1 (helper) + Task 7 (differential oracle). ✓
- §3 PULL PSD repair + buffer → Task 2. ✓
- §4 per-dim `σ_obs` → Task 4 (layout) + Task 5 (data term). ✓
- §5 divergence-guard honesty → Task 3. ✓
- §6 test hardening → Task 8 (train!→MS, coverage de-tautology, SparseGP PULL oracle) + Task 5 (σ_obs band) + Task 3 (guard test). ✓

**Placeholder scan:** No TBD/TODO. Empirical tolerances (Task 7 bands, Task 5 σ_obs band) carry concrete starting values plus an explicit "adjust via @info, keep the conjunction" instruction — a legitimate TDD step, not a placeholder.

**Type consistency:** `uvar :: u -> Vector` (per-output) is produced by `field_rhs` (Task 6) and consumed by `shooting_data_term`'s `trace .+= uvar(col)` (Task 5) — lengths match `outputdim`. `logσ_obs` is a `Vector` from Task 4's `unpack`/`nhyp` and consumed as `logσ_obs[j]` in Task 5. `pathwise_moments` returns `(μs, Σs)` (Task 1) consumed by `coverage` in Tasks 7/8. `_project_psd` (Task 2) returns a `Matrix` consumed by `pull_propagate`. `svgp_var` (Task 6) signature `(prior, Z, L_ZZ, L_S, u)` matches its call in the SVGP `uvar`. Consistent.

**Ordering safety:** Task 4 leaves the ext temporarily referencing the old layout; Step 7 of Task 4 explicitly defers the full-suite run to Task 5 Step 6. This is the only point where the suite is intentionally not green mid-task, and it is called out.

## Deferred (documented in spec, not implemented here)
- State-space SVGP ELBO (propagate field variance through solver sensitivity).
- Smooth divergence barrier.
- PULL for CompositeField.
- All Spec 2 (ergonomic API) items.
