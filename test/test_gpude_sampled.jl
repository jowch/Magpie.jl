# Sampled-ELBO FAST gates: FD-vs-Mooncake gradients, determinism, estimator consistency,
# solver robustness. No heavy training — the calibration/recovery TRAINING tests live in
# test_gpude_sampled_calib.jl (split out so neither file balloons in time/memory).
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
        sol = solve(
            ODEProblem(f!, [1.0], (0.0, 1.0), pf), Tsit5();
            saveat = ts, sensealg = ext.DEFAULT_SENSEALG
        )
        A = Array(sol)
        return sum(abs2, A .- X)
    end
    v0 = copy(field.v0)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    @test all(isfinite, g_mc)
    @test norm(g_mc .- g_fd) / max(norm(g_fd), Base.eps()) < 5.0e-3
    # determinism: same ε ⇒ identical loss
    @test loss(v0) == loss(v0)
end

@testset "Task 2: SVGP+SingleShooting sampled ELBO — grad, determinism" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    a = -0.5
    ts = collect(range(0.0, 2.0; length = 12))
    X = reshape(exp.(a .* ts), 1, 12)
    Z0 = [[x] for x in range(0, 1; length = 4)]
    field = SVGPField(SqExponentialKernel(), Z0; dout = 1)
    loss = ext.svgp_sampled_loss(field, [(ts, X)], Magpie.SingleShooting(), (0.0, 2.0); nsamples = 4, seed = 3)
    v0 = copy(field.v0)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    @test all(isfinite, g_mc)
    @test norm(g_mc .- g_fd) / max(norm(g_fd), Base.eps()) < 5.0e-3
    @test loss(v0) == loss(v0)                              # deterministic (frozen ε)
end

@testset "Task 2: train! SVGP+SingleShooting runs end-to-end with nsamples" begin
    rng = MersenneTwister(4)
    a = -0.4
    ts = collect(range(0.0, 2.0; length = 20))
    X = reshape(exp.(a .* ts) .+ 0.02 .* randn(rng, 20), 1, 20)
    field = SVGPField(Magpie._kernel(0.0, 0.0), [[x] for x in range(0, 1; length = 5)]; dout = 1)
    ret = train!(field, (ts, X); nsamples = 8, adam_iters = 200, maxiters = 60)
    @test ret === field
    @test all(isfinite, field.v0)
end

@testset "Task 4: SVGP+MultipleShooting — grad incl ∂s0 live (FD-matched)" begin
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
    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), Base.eps())
    @info "Task 4 grad gate" relerr
    @test relerr < 5.0e-3
    s2 = (off + 2):(off + 2)                            # a free s0 node entry
    @test norm(g_fd[s2]) > 1.0e-5                       # ∂loss/∂s0 live
end

@testset "Task 6: sampled ELBO is a consistent estimator (variance shrinks ∝ 1/S)" begin
    # Proxy for spec §5.3 (linear-field equivalence): the sampled data term is an unbiased MC
    # estimate, so its spread across independent seeds shrinks as S grows (variance ∝ 1/S). At the
    # untrained init the variational posterior is the (wide) prior, so the per-seed spread is large
    # at small S and visibly tighter at large S — the defining property of a consistent estimator.
    # (We check consistency rather than reconstructing the removed analytic mean-field+trace objective.)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    a = -0.5; ts = collect(range(0.0, 2.0; length = 12)); X = reshape(exp.(a .* ts), 1, 12)
    field = SVGPField(SqExponentialKernel(), [[x] for x in range(0, 1; length = 4)]; dout = 1)
    v = copy(field.v0)
    mk(S, seed) = ext.svgp_sampled_loss(field, [(ts, X)], Magpie.SingleShooting(), (0.0, 2.0); nsamples = S, seed = seed)(v)
    relspread(S) = (vals = [mk(S, s) for s in 1:8]; (maximum(vals) - minimum(vals)) / abs(sum(vals) / 8))
    sp8 = relspread(8); sp64 = relspread(64)
    @info "estimator spread" sp8 sp64
    @test sp64 < sp8          # 8× more samples ⇒ ≈2.8× tighter spread (variance ∝ 1/S)
end

@testset "Task 6: sampled-field divergence → finite sentinel (solver robustness, spec §5.6)" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    field = SVGPField(SqExponentialKernel(), [[x] for x in range(-1, 1; length = 4)]; dout = 1)
    ts = collect(range(0.0, 1.0; length = 8)); X = reshape(exp.(-0.5 .* ts), 1, 8)
    loss = ext.svgp_sampled_loss(field, [(ts, X)], Magpie.SingleShooting(), (0.0, 1.0); nsamples = 4, seed = 5)
    vbad = copy(field.v0); vbad[2] = 20.0       # logσ=20 ⇒ σ²≈e40 ⇒ field blows up ⇒ ODE diverges
    Lbad = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        loss(vbad)
    end
    @test isfinite(Lbad)                         # finite sentinel, not NaN/Inf/throw
    @test Lbad ≥ 1.0e5                           # divergence guard fired (1e6 sentinel)
end
