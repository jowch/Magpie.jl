# SVGP-UDE Sampled-ELBO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SVGP through-solver mean-field+local-trace objective with a sampled reparameterized (Matheron) ELBO, and on it enable SVGP+MultipleShooting and CompositeField+SVGP-residual, all to the ExactGP path's correctness standard.

**Architecture:** A single new primitive — `svgp_sample_rhs(field, v, ε_s)` builds one differentiable Matheron-sample RHS with trained params threaded through `pf` (R1). The sampled loss loops `S` frozen-ε samples over the **existing field-agnostic `shooting_data_term`** with the trace term OFF (variance comes from sampling), averages the per-sample data terms, and adds KL+priors once. SingleShooting and MultipleShooting reuse the same engine (MS adds only `s0` packing); CompositeField passes `known` through the existing `known_physics` kwarg.

**Tech Stack:** Julia, AbstractGPs/KernelFunctions, OrdinaryDiffEq + SciMLSensitivity (`GaussAdjoint`+`MooncakeVJP`), DifferentiationInterface + Mooncake (outer AD), Optimization.jl (Adam→LBFGS).

## Global Constraints

- **Objective (spec §2):** `L(v) = −(1/S)Σ_s log p(X|states(f_s)) + Σ_i KL(q_i‖p_i) + priors`. Per-sample data term is the **pure Gaussian NLL** (`_gaussian_nll`) — NO local-trace correction (pass an all-zero `uvar`/`trace=false`).
- **Reparameterization:** ε = (ε_ω, ε_b, ε_w, ε_ind) indexed by (sample s, output i), **frozen for the whole optimization run** (drawn once at loss construction, captured). Loss must be deterministic in `v`. Antithetic pairing (`ε` and `−ε`); `nsamples` even.
- **R1:** never closure-capture trained params (`logℓ,logσ,Z,μ,L_S`) into an ODE RHS; thread them through `pf`. Only the frozen ε and `b` are constants and may be captured.
- **R2:** `Array(sol)` before indexing.
- **Jitter:** relative `field.jitter·σ²` on `K_ZZ` (Mooncake Cholesky-backward safety). All factorizations via the `_chol` chokepoint (`cholesky(Symmetric(·); check=false)`).
- **Mooncake-clean:** dense matrix ops only on the AD path (no `LowerTriangular` backsolve, no `max(0,·)` clamp); convert factors to `Matrix` for closures as the existing SVGP `field_rhs` does.
- **Test gate:** every SciML/UDE test runs only under `MAGPIE_TEST_SCIML=true`. Verify with `MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'` (~13–18 min). Direct `julia --project=. test/<file>.jl` does NOT work for SciML files (OrdinaryDiffEq/SciMLSensitivity are test-only deps). Run the full suite as a SINGLE BLOCKING foreground call; do not poll.
- **Default `nsamples=16`.**
- **ExactGP path is untouched** — only SVGP-inner training changes.

---

### Task 1: Frozen-ε reparameterization + single-sample Matheron RHS primitive

**Files:**
- Modify: `src/gpude.jl` (add `_svgp_sample_eps`, helpers near the decoupled-sampler section ~`:438`)
- Modify: `ext/MagpieSciMLExt.jl` (add `svgp_sample_rhs` near `field_rhs(::SVGPField)` ~`:113`)
- Test: `test/test_gpude_sampled.jl` (new; add `include` to `test/runtests.jl` under the `MAGPIE_TEST_SCIML` block)

**Interfaces:**
- Produces:
  - `Magpie._svgp_sample_eps(field::SVGPField, nsamples::Int; seed::Int) -> Vector{NamedTuple}` — one entry per sample with fields `ω::Matrix` (D×Drff), `b::Vector` (Drff), `w::Vector` (Drff), `ind::Matrix` (M×dout); antithetic (second half = negation of first half). `Drff` is a field/const (default 256).
  - `svgp_sample_rhs(field::SVGPField, v, e) -> (pf, rhs!, b)` — `pf = vcat(logℓ, logσ, vec(ω), w, vec(Z), vec(v_corr))` where `ω=ε_ω/ℓ`, `w=σ·ε_w`, `u_s=L_ZZ(μ_i+L_{S,i}·ε_ind_i)`, `v_corr_i = K_ZZ⁻¹(u_s−Φw)` (dout columns); `rhs!(du,u,pf,t; known_physics)` does `du .= known_physics(u,t)` then adds, per output i, `wᵀφ(u) + Σⱼ k(u,Zⱼ)·v_corr[j,i]`. `b` (Drff) is the captured RFF phase constant.
- Consumes: existing `Magpie.svgp_Z/svgp_μ/svgp_Lsblk/unpack_LS/_chol/_kernel`, `field.M/dout/D/jitter`.

- [ ] **Step 1: Write the failing test** — `test/test_gpude_sampled.jl`:

```julia
using Test, Magpie, LinearAlgebra, Random
using OrdinaryDiffEq, SciMLSensitivity, KernelFunctions
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences

@testset "Task 1: single Matheron-sample RHS differentiates through solver (FD-matched)" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    Z0 = [[x] for x in range(-1, 1; length = 4)]
    field = SVGPField(SqExponentialKernel(), Z0; dout = 1)
    eps = Magpie._svgp_sample_eps(field, 2; seed = 1)
    ts = collect(range(0.0, 1.0; length = 8)); X = reshape(exp.(-0.5 .* ts), 1, 8)
    function loss(v)
        pf, rhs!, b = ext.svgp_sample_rhs(field, v, eps[1])
        f!(du, u, p, t) = rhs!(du, u, p, t; known_physics = (u, t) -> zero(u))
        sol = solve(ODEProblem(f!, [1.0], (0.0, 1.0), pf), Tsit5();
                    saveat = ts, sensealg = ext.DEFAULT_SENSEALG)
        A = Array(sol)
        return sum(abs2, A .- X)
    end
    v0 = copy(field.v0)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    @test all(isfinite, g_mc)
    @test norm(g_mc .- g_fd) / max(norm(g_fd), eps()) < 5.0e-3
    # determinism: same ε ⇒ identical loss
    @test loss(v0) == loss(v0)
end
```

- [ ] **Step 2: Run to verify it fails** — `MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'` (or, faster during dev, a temp env including the file). Expect: `svgp_sample_rhs` / `_svgp_sample_eps` undefined.

- [ ] **Step 3: Implement `_svgp_sample_eps` in `src/gpude.jl`** — antithetic frozen noise (validated shape from the feasibility spike):

```julia
# Frozen reparameterization noise for the sampled ELBO. Antithetic: second half negates first.
# Drff random Fourier features (default 256). One NamedTuple per sample; ind is M×dout (per output).
const SVGP_DRFF = 256
function _svgp_sample_eps(field::SVGPField, nsamples::Int; seed::Int = 0)
    @assert iseven(nsamples) "nsamples must be even (antithetic pairing)"
    half = nsamples ÷ 2
    base = map(1:half) do s
        r = Random.MersenneTwister(seed * 100_003 + s)
        (
            ω = randn(r, field.D, SVGP_DRFF),
            b = rand(r, SVGP_DRFF) .* (2π),         # phase is NOT antithetic (shared with its pair)
            w = randn(r, SVGP_DRFF),
            ind = randn(r, field.M, field.dout),
        )
    end
    anti = map(base) do e
        (ω = -e.ω, b = e.b, w = -e.w, ind = -e.ind)   # negate Gaussian draws; keep phase b
    end
    return vcat(base, anti)
end
```
Export-free (internal). Add `svgp_sample_rhs` to the `using Magpie: …` import list consumed by the ext if needed; `_svgp_sample_eps` stays `Magpie._svgp_sample_eps`.

- [ ] **Step 4: Implement `svgp_sample_rhs` in `ext/MagpieSciMLExt.jl`** (adapt the validated spike; dense Mooncake-safe ops, relative jitter, R1 threading):

```julia
function svgp_sample_rhs(field::SVGPField, v, e)
    M, dout, D = field.M, field.dout, field.D
    logℓ, logσ = v[1], v[2]
    ℓ = exp(logℓ); σ = exp(logσ); σ2 = σ^2
    k = Magpie._kernel(logℓ, logσ)
    Z = Magpie.svgp_Z(field, v); Zvec = [Z[:, j] for j in 1:M]
    ω = e.ω ./ ℓ; w = σ .* e.w; b = e.b
    rff(x) = sqrt(2 / size(ω, 2)) .* cos.(ω' * x .+ b)
    jit = field.jitter * σ2
    C = _chol(kernelmatrix(k, Zvec) + jit * I)
    LZZ = Matrix(C.L)
    μ = Magpie.svgp_μ(field, v)                                   # M×dout
    Φw = [dot(w, rff(Zvec[j])) for j in 1:M]
    vcorr = Matrix{eltype(v)}(undef, M, dout)
    for i in 1:dout
        Ls = Matrix(Magpie.unpack_LS(Magpie.svgp_Lsblk(field, v, i), M))
        u_s = LZZ * (μ[:, i] .+ Ls * e.ind[:, i])
        vcorr[:, i] = C \ (u_s .- Φw)
    end
    pf = vcat(logℓ, logσ, vec(ω), w, vec(Z), vec(vcorr))
    function rhs!(du, u, _pf, t; known_physics)
        du .= known_physics(u, t)
        ω_r = reshape(_pf[3:(2 + D * size(ω, 2))], D, size(ω, 2))
        o = 2 + D * size(ω, 2)
        w_r = _pf[(o + 1):(o + size(ω, 2))]; o += size(ω, 2)
        Z_r = reshape(_pf[(o + 1):(o + D * M)], D, M); o += D * M
        vc = reshape(_pf[(o + 1):(o + M * dout)], M, dout)
        kk = Magpie._kernel(_pf[1], _pf[2])
        prior = dot(w_r, sqrt(2 / size(ω, 2)) .* cos.(ω_r' * u .+ b))
        for i in 1:dout
            du[i] += prior + sum(kk(u, @view Z_r[:, j]) * vc[j, i] for j in 1:M)
        end
        return nothing
    end
    return (pf, rhs!, b)
end
```
Note: `prior` is shared across outputs here for brevity — CORRECT only if outputs share the RFF draw. Per spec, each output has its OWN prior draw. Fix: make `ω/w/Φw/rff` per-output (index `e.ω[:,:,i]`…) OR give `e` per-output RFF. **Decision:** give `_svgp_sample_eps` per-output RFF: `ω::Array(D,Drff,dout)`, `w::Matrix(Drff,dout)`, and loop the prior per output. Implementer: thread per-output RFF so each output dim has an independent prior sample (matches `_svgp_pathwise_sampler`, which draws per output).

- [ ] **Step 5: Run the test to verify it passes** — full gated suite. Expect the Task-1 testset green (FD-matched, deterministic).

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(svgp): single Matheron-sample RHS primitive (frozen-ε, pf-threaded, FD-matched)"`

---

### Task 2: Sampled-ELBO loss + SVGP+SingleShooting replacement

**Files:**
- Modify: `ext/MagpieSciMLExt.jl` (rewrite `svgp_elbo_loss` as the sampled loop; update `_train_loss(::SVGPField, ::SingleShooting)`; add `nsamples` to `train!`)
- Test: `test/test_gpude_sampled.jl`

**Interfaces:**
- Produces: `svgp_sampled_loss(field::SVGPField, trajs, shooting, tspan; nsamples=16, seed=0, kw...) -> (v->Real)` — loops `nsamples` frozen-ε samples, each calling the existing `shooting_data_term(field, shooting, (pf_s,rhs!_s), data; logσ_obs=v[3:2+dout], uvar=(_->zeros(dout)), _ms_kwargs(field,shooting,v,dout)..., kw...)`, averages, adds `Magpie.regularizer(field, v; kw...)` once. `svgp_elbo_loss` becomes a thin alias to it (SingleShooting).
- Consumes: Task 1 `svgp_sample_rhs`, `_svgp_sample_eps`; existing `shooting_data_term`, `regularizer`, `_ms_kwargs`.

- [ ] **Step 1: Write the failing tests** (append to `test/test_gpude_sampled.jl`):

```julia
@testset "Task 2: SVGP+SingleShooting sampled ELBO — grad, determinism, linear equivalence" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    a = -0.5; ts = collect(range(0.0, 2.0; length = 12)); X = reshape(exp.(a .* ts), 1, 12)
    Z0 = [[x] for x in range(0, 1; length = 4)]
    field = SVGPField(SqExponentialKernel(), Z0; dout = 1)
    loss = ext.svgp_sampled_loss(field, [(ts, X)], Magpie.SingleShooting(), (0.0, 2.0); nsamples = 4, seed = 3)
    v0 = copy(field.v0)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    @test all(isfinite, g_mc)
    @test norm(g_mc .- g_fd) / max(norm(g_fd), eps()) < 5.0e-3
    @test loss(v0) == loss(v0)                              # deterministic (frozen ε)
end

@testset "Task 2: train! SVGP+SingleShooting runs end-to-end with nsamples" begin
    rng = MersenneTwister(4); a = -0.4
    ts = collect(range(0.0, 2.0; length = 20)); X = reshape(exp.(a .* ts) .+ 0.02 .* randn(rng, 20), 1, 20)
    field = SVGPField(Magpie._kernel(0.0, 0.0), [[x] for x in range(0, 1; length = 5)]; dout = 1)
    ret = train!(field, (ts, X); nsamples = 8, adam_iters = 200, maxiters = 60)
    @test ret === field
    @test all(isfinite, field.v0)
end
```

- [ ] **Step 2: Run to verify failure** — `svgp_sampled_loss` undefined / `nsamples` kwarg unknown.

- [ ] **Step 3: Implement `svgp_sampled_loss`** (replace the `svgp_elbo_loss` body, `ext/MagpieSciMLExt.jl:420–424`):

```julia
function svgp_sampled_loss(field::SVGPField, trajectories, shooting, tspan; nsamples::Int = 16, seed::Int = 0, kw...)
    dout = field.dout
    eps = Magpie._svgp_sample_eps(field, nsamples; seed)
    return function (v)
        acc = zero(eltype(v))
        for e in eps
            pf, rhs!, _ = svgp_sample_rhs(field, v, e)
            acc += shooting_data_term(
                field, shooting, (pf, rhs!), trajectories;
                logσ_obs = v[3:(2 + dout)], uvar = (_ -> zeros(dout)),
                _ms_kwargs(field, shooting, v, dout)..., tspan = tspan, kw...
            )
        end
        return acc / length(eps) + Magpie.regularizer(field, v; kw...)
    end
end
# back-compat thin alias (SingleShooting)
svgp_elbo_loss(field::SVGPField, trajectories; tspan, nsamples = 16, kw...) =
    svgp_sampled_loss(field, trajectories, Magpie.SingleShooting(), tspan; nsamples, kw...)
```

- [ ] **Step 4: Wire `nsamples` through `train!` and `_train_loss`** (`:320`, `:354`): add `nsamples = 16` kwarg to `Magpie.train!`; change `_train_loss(field::SVGPField, trajs, ::SingleShooting, tspan; kw...)` to `svgp_sampled_loss(field, trajs, Magpie.SingleShooting(), tspan; kw...)` (nsamples arrives via `kw...`). Ensure `nsamples`/`seed` are NOT forwarded into `shooting_data_term`/`regularizer` in a way that errors (they consume `; kw...` and ignore extras — verify).

- [ ] **Step 5: Run the gated suite** — Task-2 testsets green; existing ExactGP tests unaffected. (SVGP calibration/noise may now FAIL — expected; fixed in Task 3.)

- [ ] **Step 6: Commit** — `feat(svgp): sampled-ELBO loss replaces mean-field SingleShooting path`

---

### Task 3: Re-derive SVGP SingleShooting test expectations + nonlinear calibration gate (SS)

**Files:**
- Modify: `test/test_gpude_calibration.jl`, `test/test_gpude_noise.jl`
- Test: `test/test_gpude_sampled.jl` (add the nonlinear LV calibration gate)

- [ ] **Step 1: Run the suite, record actual SVGP calibration/noise numbers** under the sampled ELBO (the old mean-field expectations are now stale). Capture cov/width/σ_obs values from `@info` lines.

- [ ] **Step 2: Re-derive `test_gpude_calibration.jl`** — the trace-ON/OFF contrast is obsolete (no trace term now; `trace` kwarg no longer governs SVGP variance). Replace with: train one SVGP via sampled ELBO, assert held-out cov90 is near nominal (`abs(cov - 0.9) < 0.15`) and intervals are finite/sharp (width below the prior-variance ceiling). Keep the 1-D linear field.

- [ ] **Step 3: Re-derive `test_gpude_noise.jl`** — keep the σ_obs-recovery assertion but regenerate the expected band under the sampled objective; update the direct-loss-call site (`:94`) to `svgp_sampled_loss(...; nsamples=…)`. Keep the per-dim σ_obs gradient-liveness FD test (update if the loss entrypoint changed).

- [ ] **Step 4: Add the nonlinear (LV) SS calibration gate** to `test/test_gpude_sampled.jl`:

```julia
@testset "Task 3: SVGP+SingleShooting calibrated on nonlinear LV (cov90 ≈ 0.9)" begin
    lv!(du, u, p, t) = (du[1] = 1.5u[1] - u[1]*u[2]; du[2] = u[1]*u[2] - 3u[2]; nothing)
    u0 = [1.0, 1.0]; tspan = (0.0, 2.5); ts = collect(range(tspan...; length = 60))
    clean = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat = ts, abstol = 1e-9, reltol = 1e-9))
    X = clean .+ 0.02 .* randn(MersenneTwister(1), size(clean))
    field = SVGPField(SqExponentialKernel(), kmeans_anchors(X, 15; rng = MersenneTwister(7)); dout = 2)
    train!(field, (ts, X); nsamples = 16, adam_iters = 600, maxiters = 150)
    ens = propagate(field, u0, tspan; method = Pathwise(128), ts = ts)
    μ, Σ = pathwise_moments(ens)
    cov90 = coverage([collect(clean[:, k]) for k in 1:length(ts)], μ, Σ; level = 0.9)
    @info "SVGP SS LV calibration" cov90
    @test abs(cov90 - 0.9) < 0.2          # nonlinear regime; the real correctness gate
end
```
(Tune `nsamples`/iters if the band is borderline; the gate is calibration near nominal, not a knife-edge.)

- [ ] **Step 5: Run the gated suite** — all SVGP SingleShooting tests green under the new objective.

- [ ] **Step 6: Commit** — `test(svgp): re-derive SS expectations + nonlinear calibration gate`

---

### Task 4: SVGP + MultipleShooting

**Files:**
- Modify: `ext/MagpieSciMLExt.jl` (`_train_loss`/`_train_init` for `(::SVGPField, ::MultipleShooting)`; remove the SVGP+MS guard `:340–341`)
- Test: `test/test_gpude_sampled.jl`, `test/test_gpude_stage2.jl` (replace the "SVGP+MS fails clearly" block)

**Interfaces:**
- Produces: `_train_loss(::SVGPField, trajs, ms::MultipleShooting, tspan; kw...) = svgp_sampled_loss(field, trajs, ms, tspan; kw...)`; `_train_init(::SVGPField, trajs, ms::MultipleShooting)` = `_init_vec(field, u_data, t_data, ms)` (appends data-seeded `s0` after `field.v0`). The per-sample `shooting_data_term(::MultipleShooting)` returns NLL+continuity+anchor PER SAMPLE; the sample-loop averaging in `svgp_sampled_loss` averages the penalties too (spec §3).

- [ ] **Step 1: Write failing tests** (append to `test/test_gpude_sampled.jl`):

```julia
@testset "Task 4: SVGP+MultipleShooting — grad incl ∂s0, runs e2e" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    a = -0.3; ts = collect(range(0.0, 6.0; length = 24)); X = reshape(exp.(a .* ts), 1, 24)
    field = SVGPField(SqExponentialKernel(), [[x] for x in range(0, 2; length = 4)]; dout = 1)
    ms = MultipleShooting(nsegments = 4)
    trajs = [(ts, X)]
    loss = ext.svgp_sampled_loss(field, trajs, ms, (0.0, 6.0); nsamples = 4, seed = 2)
    off = length(field.v0)
    s0 = hcat([X[:, round(Int, i)] for i in range(1, 24; length = 4)]...)
    v = vcat(copy(field.v0), vec(s0))
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v)[1]
    @test norm(g_mc .- g_fd) / max(norm(g_fd), eps()) < 5.0e-3
    s2 = (off + 1 + 1):(off + 2)                       # a free s0 node entry
    @test norm(g_fd[s2]) > 1.0e-5                       # ∂loss/∂s0 live
end

@testset "Task 4: train! SVGP+MultipleShooting recovers (LV, long horizon)" begin
    lv!(du, u, p, t) = (du[1] = 1.5u[1] - u[1]*u[2]; du[2] = u[1]*u[2] - 3u[2]; nothing)
    u0 = [1.0, 1.0]; tspan = (0.0, 6.0); ts = collect(range(tspan...; length = 60))
    clean = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat = ts, abstol = 1e-9, reltol = 1e-9))
    X = clean .+ 0.02 .* randn(MersenneTwister(2), size(clean))
    field = SVGPField(SqExponentialKernel(), kmeans_anchors(X, 15; rng = MersenneTwister(7)); dout = 2)
    train!(field, (ts, X); shooting = MultipleShooting(nsegments = 8), nsamples = 16, adam_iters = 400, maxiters = 100)
    @test all(isfinite, field.v0)
    ens = propagate(field, u0, tspan; method = Pathwise(128), ts = ts)
    μ, Σ = pathwise_moments(ens)
    cov90 = coverage([collect(clean[:, k]) for k in 1:length(ts)], μ, Σ; level = 0.9)
    @info "SVGP MS LV calibration" cov90
    @test abs(cov90 - 0.9) < 0.25
end
```

- [ ] **Step 2: Run to verify failure** — currently the SVGP+MS guard throws `ArgumentError`.

- [ ] **Step 3: Remove the guard + add the MS methods** — delete `_assert_shooting_supported(::SVGPField, ::Magpie.MultipleShooting)` (`:340–341`); add:
```julia
_train_loss(field::SVGPField, trajs, ms::Magpie.MultipleShooting, tspan; kw...) =
    svgp_sampled_loss(field, trajs, ms, tspan; kw...)
_train_init(field::SVGPField, trajs, ms::Magpie.MultipleShooting) =
    (tu = only(trajs); _init_vec(field, tu[2], tu[1], ms))
```
`_ms_kwargs(field, ms, v, dout)` already reads `s0` at `offset = length(field.v0)` — correct for SVGP since `field.v0` is the full SVGP prefix. `train!` already drops the `s0` tail when storing (`field.v0 .= sol.u[1:length(field.v0)]`).

- [ ] **Step 4: Replace the stage2 "SVGP+MS fails clearly" block** (`test/test_gpude_stage2.jl:109–116`) — the SVGP+MS half now WORKS; keep only the still-unsupported guard if any remain (Composite+SVGP becomes supported in Task 5, so this whole `@testset` is removed once Task 5 lands; for Task 4 keep the Composite+SVGP `@test_throws` half, drop the SVGP+MS half).

- [ ] **Step 5: Run the gated suite** — Task-4 testsets green; gradient FD-matched; ∂s0 live.

- [ ] **Step 6: Commit** — `feat(svgp): MultipleShooting via sampled ELBO (s0 packing, guard removed)`

---

### Task 5: CompositeField + SVGP residual

**Files:**
- Modify: `ext/MagpieSciMLExt.jl` (`_train_loss` for CompositeField-with-SVGP-inner; remove the composite guard `:345–348`; fix `_pathwise_composite` dispatch for SparseGP inner)
- Test: `test/test_gpude_sampled.jl`; update the matrix tests in `test/test_gpude_protocol.jl` and `test/test_gpude_stage2.jl`

**Interfaces:**
- Produces:
  - `_train_loss(cf::CompositeField, trajs, shooting, tspan; kw...)` — when `cf.gp isa SVGPField`, route to `svgp_sampled_loss(cf.gp, trajs, shooting, tspan; known_physics = cf.known, kw...)`; otherwise the existing `field_loss(cf, …)` path (Exact inner, unchanged).
  - `_pathwise_composite(gps::AbstractVector{<:SparseGP}, known, u0, tspan, ts, m)` = `_pathwise_integrate(gps, _svgp_pathwise_sampler, u0, tspan, ts, m; known)`. (`gpfield(::SVGPField)` is NOT added — prediction flows through `posterior(cf.gp,v)→SparseGP→svgp_moments`; the only gap was the composite Pathwise sampler.)

- [ ] **Step 1: Write failing tests** (append to `test/test_gpude_sampled.jl`):

```julia
@testset "Task 5: Composite(SVGP) trains + recovers residual" begin
    known(u, t) = -0.5 .* u
    f!(du, u, p, t) = (du .= known(u, t); du .+= 0.3; nothing)
    u0 = [1.0]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 40))
    target = Array(solve(ODEProblem(f!, u0, tspan), Tsit5(); saveat = ts))
    X = target .+ 0.02 .* randn(MersenneTwister(11), size(target))
    cf = CompositeField(known, SVGPField(SqExponentialKernel(), kmeans_anchors(X, 6; rng = MersenneTwister(3)); dout = 1))
    ret = train!(cf, (ts, X); nsamples = 16, adam_iters = 400, maxiters = 100)
    @test ret === cf
    res = predmean(posterior(cf)[1], [0.5])           # residual GP mean (SparseGP)
    @test isfinite(res)
    @test abs(res - 0.3) < 0.3                          # recovers the +0.3 residual, not 0
    ens = propagate(cf, u0, tspan; method = Pathwise(64), ts = ts)   # SparseGP composite Pathwise
    @test size(ens) == (64, 1, length(ts))
end
```

- [ ] **Step 2: Run to verify failure** — composite guard throws; composite Pathwise would hit `_exact_pathwise_sampler` on a `SparseGP`.

- [ ] **Step 3: Remove the composite guard + route SVGP-inner training** — delete `_assert_shooting_supported(cf::CompositeField, …)` body's SVGP branch (`:345–348`); make `_train_loss(cf, …)` dispatch on `cf.gp isa SVGPField` to `svgp_sampled_loss(cf.gp, …; known_physics = cf.known, kw...)`. `_train_init(cf, …)` already delegates to the inner gp (works for SVGP).

- [ ] **Step 4: Fix `_pathwise_composite` dispatch** — add the `SparseGP` method (above) alongside the existing ExactGP one.

- [ ] **Step 5: Update the matrix tests** — `test/test_gpude_protocol.jl` (the Composite SVGP-inner row: was `@test_throws ArgumentError train!`, now trains + `predmean(posterior(cf_sv)[1], …)` finite); remove the now-empty "fails clearly" block in `test/test_gpude_stage2.jl`.

- [ ] **Step 6: Run the gated suite + commit** — `feat(svgp): CompositeField+SVGP residual (sampled ELBO, Pathwise dispatch)`

---

### Task 6: Remaining correctness gates + hygiene

**Files:**
- Test: `test/test_gpude_sampled.jl`, `test/test_gpude_svgp.jl`
- Modify: `ext/MagpieSciMLExt.jl` (remove the now-dead `_assert_shooting_supported` default if no guards remain — keep the generic `= nothing` method), `CLAUDE.md` (status), `src/Magpie.jl` (only if any export changed — expected: none)

- [ ] **Step 1: Linear-field equivalence gate** — in `test/test_gpude_sampled.jl`: for a linear field `du=Au`, assert the sampled data term (large `S`, e.g. 64) approaches the analytic mean-field+trace value (reconstruct the old objective inline at the same `v`); tolerance set by MC error (`rtol` ~ a few %). Validates the estimator against the known-correct analytic form where they must agree.

- [ ] **Step 2: M=N limit invariant** — extend `test/test_gpude_svgp.jl`: with inducing points placed at the data (M=N), the deterministic posterior (`predmean`/`var` via `svgp_moments`) matches ExactGP moments within the existing tolerance (this is a posterior/moment invariant, solver-free — confirm the existing test already covers it; tighten/relabel if needed).

- [ ] **Step 3: Solver-robustness gate** — assert that a sampled-field RHS that diverges returns the finite sentinel (loss `≥ 1e6`, not NaN/throw) and that the gradient stays finite (relative-jitter backward guard holds). Wrap any expected solver warnings in `Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger())`.

- [ ] **Step 4: Export/guard audit** — confirm `src/Magpie.jl` exports unchanged; confirm only the generic `_assert_shooting_supported(field, shooting) = nothing` remains (both specific guards removed); grep for any lingering "tracked follow-up" / "not supported" SVGP messages and remove.

- [ ] **Step 5: Update `CLAUDE.md` status** — note Capability B now supports SVGP via sampled ELBO, SVGP+MultipleShooting, and CompositeField+SVGP; the mean-field+local-trace SVGP path is replaced.

- [ ] **Step 6: Full gated suite (single blocking run) + commit** — `MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'` green; `test(svgp): linear-equivalence, M=N, solver-robustness gates + status`

---

## Self-Review notes (author)

- **Spec coverage:** §2 objective → Tasks 1–2; §3 three consumers → Tasks 2 (SS), 4 (MS), 5 (Composite); §4 API (`nsamples`, guards) → Tasks 2/4/5; §5 gates → 1 (grad+determinism), 2 (grad+determinism+linear via Task 6), 3 (calibration SS), 4 (grad+∂s0+calibration MS), 6 (linear, M=N, robustness). All six gates placed.
- **Deviation from spec §3/§5:** `gpfield(::SVGPField)` is NOT implemented (YAGNI — prediction flows through `SparseGP`); the prediction-side need is the `_pathwise_composite` SparseGP dispatch (Task 5). Flagged to the user.
- **Per-output RFF:** Task 1 Step 4 note — give each output dim an independent prior draw (per-output ε), matching `_svgp_pathwise_sampler`. Implementer must thread per-output RFF.
- **Type consistency:** `svgp_sample_rhs` ↔ `svgp_sampled_loss` ↔ `shooting_data_term` signatures aligned; `_ms_kwargs` offset (`length(field.v0)`) reused for SVGP MS.
