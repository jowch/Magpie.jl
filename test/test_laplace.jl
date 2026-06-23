using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, DifferentiationInterface, Test
using Magpie: LaplaceGP
import Mooncake
@testset "LaplaceGP" begin
    # NOTE: use yb = first.(X) .> -0.5 (not .> 0.0) to avoid symmetric cancellation in
    # sum(mean(g, X)), which would make the gradient identically zero at all logℓ.
    X = [[x] for x in range(-2, 2; length = 12)]; yb = first.(X) .> -0.5
    g = Magpie.update(LaplaceGP(with_lengthscale(SqExponentialKernel(), 1.0)), X, yb)
    μ, v = mean(g, X), var(g, X)
    @test all(μ[yb] .> 0)     # positive latent mean where y=1
    @test all(μ[.!yb] .< 0)    # negative latent mean where y=0
    @test all(v .> 0)
    # latent mean must equal Ks'·a (independent recomputation from stored a)
    @test mean(g, [[0.0]]) ≈ [dot(AbstractGPs.cov(g.prior, g.x, [[0.0]]), g.a)] rtol = 1.0e-8
    loss(logℓ) = sum(mean(Magpie.update(LaplaceGP(with_lengthscale(SqExponentialKernel(), exp(only(logℓ)))), X, yb), X))
    gmc = only(DifferentiationInterface.gradient(loss, AutoMooncake(; config = nothing), [0.0]))
    gfd = (loss([1.0e-5]) - loss([-1.0e-5])) / 2.0e-5
    @test gmc ≈ gfd rtol = 1.0e-3
end

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
