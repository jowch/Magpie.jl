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
