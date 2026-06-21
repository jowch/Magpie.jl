using Magpie, AbstractGPs, KernelFunctions, Random, Test
using Magpie: ExactGP, fit, nlml, _lengthscale, _outputscale
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

@testset "fit also tunes the signal variance σ_f²" begin
    # A large-amplitude signal: a unit-variance prior cannot explain it. fit must recover an
    # output scale near the true marginal variance (~25), not leave it pinned at 1. Without this,
    # derivative/straddle acquisitions are miscalibrated on any function whose scale ≠ 1.
    Random.seed!(2)
    ktrue = 25.0 * with_lengthscale(SqExponentialKernel(), 0.7)
    X = [[x] for x in range(-3, 3; length=50)]
    y = rand(AbstractGPs.GP(ktrue)(X, 1e-4))
    g0 = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 1.0); noise=1e-4), X, y)
    @test _outputscale(g0.prior.kernel) ≈ 1.0           # starts at unit variance
    g  = fit(g0; restarts=3)
    @test 5.0 < _outputscale(g.prior.kernel) < 125.0    # recovered within a factor ~5 of truth
    @test nlml(g) ≤ nlml(g0) + 1e-6
end
