using Magpie, AbstractGPs, KernelFunctions, DifferentiationInterface, Test
using Magpie: ExactGP, nlml
import Mooncake
@testset "NLML gradient: Mooncake == finite diff" begin
    X = [[x] for x in range(0, 1; length = 10)]; y = sinpi.(first.(X))
    loss(logℓ) = nlml(Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), exp(only(logℓ))); noise = 1.0e-4), X, y))
    g_mc = only(DifferentiationInterface.gradient(loss, AutoMooncake(; config = nothing), [0.0]))
    g_fd = (loss([1.0e-6]) - loss([-1.0e-6])) / 2.0e-6              # hand-rolled central difference
    @test g_mc ≈ g_fd rtol = 1.0e-4
end

@testset "two-parameter penalized fit loss: Mooncake == finite diff" begin
    X = [randn(2) for _ in 1:12]; y = [sum(abs2, xi) for xi in X]
    μ0, σ0 = 0.0, 0.75                                       # the :auto prior centre/width
    loss(p) = nlml(Magpie.update(ExactGP(exp(p[2]) * with_lengthscale(SqExponentialKernel(), exp(p[1])); noise = 1.0e-6), X, y)) +
        0.5 * ((p[1] - μ0) / σ0)^2
    p = [0.3, 0.1]
    g_mc = DifferentiationInterface.gradient(loss, AutoMooncake(; config = nothing), p)
    h = 1.0e-6
    g_fd = [(loss(p .+ h .* (1:2 .== i)) - loss(p .- h .* (1:2 .== i))) / 2h for i in 1:2]
    @test g_mc ≈ g_fd rtol = 1.0e-4
end
