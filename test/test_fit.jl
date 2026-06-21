using Magpie, AbstractGPs, KernelFunctions, Random, Test
using Magpie: ExactGP, fit, nlml, _lengthscale
@testset "fit recovers a known lengthscale" begin
    Random.seed!(1)
    ktrue = with_lengthscale(SqExponentialKernel(), 0.5)
    X = [[x] for x in range(-3, 3; length = 40)]
    y = rand(AbstractGPs.GP(ktrue)(X, 1.0e-4))                  # X is vector-of-vectors (no obsdim depwarn)
    g0 = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 2.0); noise = 1.0e-4), X, y)
    g = fit(g0; restarts = 3)
    @test 0.25 < _lengthscale(g.prior.kernel) < 1.0
    @test nlml(g) ≤ nlml(g0) + 1.0e-6
end
