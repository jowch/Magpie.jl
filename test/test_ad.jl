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
