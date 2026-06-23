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
    X = [[x] for x in range(-3, 3; length = 50)]
    y = rand(AbstractGPs.GP(ktrue)(X, 1.0e-4))
    g0 = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 1.0); noise = 1.0e-4), X, y)
    @test _outputscale(g0.prior.kernel) ≈ 1.0           # starts at unit variance
    g = fit(g0; restarts = 3)
    @test 5.0 < _outputscale(g.prior.kernel) < 125.0    # recovered within a factor ~5 of truth
    @test nlml(g) ≤ nlml(g0) + 1.0e-6
end

@testset "fit lengthscale prior curbs small-n over-smoothing" begin
    # Fine features undersampled: sinpi(3x) (half-period ~1/3) probed at only 7 points over
    # [-1,1] (spacing ~1/3, right at the feature scale). Pure MLE can't resolve the oscillation,
    # so it explains the data with a near-flat surface and runs ℓ to the upper bound (~e^6) —
    # catastrophic over-smoothing. The :auto prior (centred on the init ℓ=0.2) holds ℓ near the
    # true feature scale instead. Stable across seeds; the gap is enormous (ratio ~1e-3).
    Random.seed!(7)
    f(x) = sinpi(3x[1])
    X = [[x] for x in range(-1, 1; length = 7)]        # very scarce, undersamples the feature
    y = f.(X) .+ 1.0e-3 .* randn(length(X))
    g0 = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.2); noise = 1.0e-3), X, y)
    g_map = fit(g0)                                  # default prior on (ℓ_prior=:auto)
    g_mle = fit(g0; ℓ_prior = nothing)                 # pure MLE
    ℓmap = _lengthscale(g_map.prior.kernel); ℓmle = _lengthscale(g_mle.prior.kernel)
    @test ℓmap < 0.5 * ℓmle                          # robust margin (actual ratio ~1e-3)
    @test ℓmap < 1.0                                 # MAP resolves the feature
    @test ℓmle > 2.0                                 # MLE ran ℓ up — over-smoothed
end

@testset "fit is kernel-generic over supported families" begin
    Random.seed!(42)
    X = [randn(2) for _ in 1:40]; y = [sum(abs2, xi) for xi in X]
    # Matérn kernels need Mooncake (ForwardDiff hits sqrt(0) at coincident points).
    # fit() with no explicit ad= should auto-select Mooncake for Matérn families.
    g52 = Magpie.update(ExactGP(with_lengthscale(Matern52Kernel(), 0.7); noise = 1.0e-3), X, y)
    fitted = Magpie.fit(g52)
    @test Magpie._basekernel(fitted.prior.kernel) isa Matern52Kernel    # family preserved, not swapped to RBF
    @test Magpie.nlml(fitted) ≤ Magpie.nlml(g52)                         # fit improved (or matched) the objective
    glin = Magpie.update(ExactGP(LinearKernel(); noise = 1.0e-4), X, y)    # unsupported family
    @test_throws ArgumentError Magpie.fit(glin)
end

@testset "fit recovers ARD (per-dimension) lengthscales" begin
    # Anisotropic target: oscillates in x1 (short ℓ), nearly flat in x2 (long ℓ).
    # fit must drive the x1 inverse-lengthscale ABOVE the x2 one.
    Random.seed!(11)
    f(x) = sinpi(2x[1]) + 0.1 * x[2]
    X = [2 .* rand(2) .- 1 for _ in 1:60]
    y = f.(X) .+ 1.0e-3 .* randn(60)
    k0 = 1.0 * with_lengthscale(SqExponentialKernel(), [0.5, 0.5])    # ARD, isotropic start
    g0 = Magpie.update(ExactGP(k0; noise = 1.0e-3), X, y)
    g = fit(g0)
    invℓ = g.prior.kernel.kernel.transform.v     # ScaledKernel → TransformedKernel → ARDTransform.v (inverse lengthscales)
    @test invℓ[1] > invℓ[2]                       # x1 lengthscale shorter (larger inverse) than x2
    @test nlml(g) ≤ nlml(g0) + 1.0e-6
end

@testset "fit handles a composite (sum) kernel" begin
    Random.seed!(12)
    X = [randn(2) for _ in 1:40]; y = [sum(abs2, xi) for xi in X]
    k0 = 1.0 * with_lengthscale(SqExponentialKernel(), 0.8) +
        1.0 * with_lengthscale(Matern32Kernel(), 0.8)
    g0 = Magpie.update(ExactGP(k0; noise = 1.0e-3), X, y)
    g = fit(g0)
    @test nlml(g) ≤ nlml(g0) + 1.0e-6                    # composite fit improved the objective
    @test g.prior.kernel isa KernelFunctions.KernelSum   # structure preserved through destructure/rebuild
end
