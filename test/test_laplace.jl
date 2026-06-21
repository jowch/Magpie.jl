using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, DifferentiationInterface, Test
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
