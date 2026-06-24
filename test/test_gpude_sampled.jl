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

@testset "Task 2: SVGP+SingleShooting sampled ELBO — grad, determinism, linear equivalence" begin
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

@testset "Task 3: SVGP+SingleShooting calibrated on nonlinear LV (cov90 ≈ 0.9)" begin
    lv!(du, u, p, t) = (du[1] = 1.5u[1] - u[1] * u[2]; du[2] = u[1] * u[2] - 3u[2]; nothing)
    u0 = [1.0, 1.0]
    tspan = (0.0, 2.5)
    ts = collect(range(tspan...; length = 60))
    clean = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat = ts, abstol = 1.0e-9, reltol = 1.0e-9))
    X = clean .+ 0.02 .* randn(MersenneTwister(1), size(clean))
    field = SVGPField(SqExponentialKernel(), kmeans_anchors(X, 15; rng = MersenneTwister(7)); dout = 2)
    train!(field, (ts, X); nsamples = 16, adam_iters = 600, maxiters = 150)
    ens = propagate(field, u0, tspan; method = Pathwise(128), ts = ts)
    μ, Σ = pathwise_moments(ens)
    cov90 = coverage([collect(clean[:, k]) for k in 1:length(ts)], μ, Σ; level = 0.9)
    @info "SVGP SS LV calibration" cov90
    @test abs(cov90 - 0.9) < 0.2          # nonlinear regime; the real correctness gate
end

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
    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    @info "Task 4 grad gate" relerr
    @test relerr < 5.0e-3
    s2 = (off + 1 + 1):(off + 2)                       # a free s0 node entry
    @test norm(g_fd[s2]) > 1.0e-5                       # ∂loss/∂s0 live
end

@testset "Task 4: train! SVGP+MultipleShooting recovers (1D decay, long horizon)" begin
    # 1D linear decay. Light budget exercises the full SVGP+MS wiring end-to-end
    # (s0 packing, guard removed, sampled-ELBO through MultipleShooting solver path).
    # nsamples=8, 8 segments, 300 ADAM + 80 LBFGS — per task brief (§IMPORTANT budgets).
    a = -0.3; u0 = [2.0]; tspan = (0.0, 6.0); ts = collect(range(tspan...; length = 36))
    clean = reshape(u0[1] .* exp.(a .* ts), 1, 36)
    X = clean .+ 0.02 .* randn(MersenneTwister(2), size(clean))
    field = SVGPField(Magpie._kernel(0.0, 0.0), [[x] for x in range(0, 2; length = 5)]; dout = 1)
    train!(field, (ts, X); shooting = MultipleShooting(nsegments = 8), nsamples = 8, adam_iters = 300, maxiters = 80)
    @test all(isfinite, field.v0)
    ens = propagate(field, u0, tspan; method = Pathwise(64), ts = ts)
    μ, Σ = pathwise_moments(ens)
    cov90 = coverage([collect(clean[:, k]) for k in 1:length(ts)], μ, Σ; level = 0.9)
    @info "SVGP MS 1D calibration" cov90
    @test abs(cov90 - 0.9) < 0.25
end

@testset "Task 5: Composite(SVGP) trains + recovers residual + Pathwise ensemble" begin
    # 1D scalar ODE: known = -0.5u, residual = +0.3 (constant offset).
    # The GP must learn to output ≈0.3 at states visited during training.
    known(u, t) = -0.5 .* u
    f!(du, u, p, t) = (du .= known(u, t); du .+= 0.3; nothing)
    u0 = [1.0]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 40))
    target = Array(solve(ODEProblem(f!, u0, tspan), Tsit5(); saveat = ts))
    X = target .+ 0.02 .* randn(MersenneTwister(11), size(target))
    cf = CompositeField(known, SVGPField(SqExponentialKernel(), kmeans_anchors(X, 6; rng = MersenneTwister(3)); dout = 1))

    # train! must not throw (guard removed) and must return the field
    ret = train!(cf, (ts, X); nsamples = 8, adam_iters = 300, maxiters = 80)
    @test ret === cf

    # posterior returns SparseGPs (SVGPField inner); predmean callable
    gps = posterior(cf)
    @test gps[1] isa Magpie.SparseGP
    res = predmean(gps[1], [0.5])
    @test isfinite(res)
    @test abs(res - 0.3) < 0.3    # recovers the +0.3 residual within tolerance

    # Pathwise ensemble exercises the SparseGP composite dispatch
    ens = propagate(cf, u0, tspan; method = Pathwise(64), ts = ts)
    @test size(ens) == (64, 1, length(ts))
    @test all(isfinite, ens)
end
