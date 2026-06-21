# test/test_derivatives.jl
using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test, ForwardDiff
using Magpie: ExactGP, grad_predict, predmean, _lengthscale, _outputscale, update
# (LinearAlgebra provides the `norm`/`eigvals`/`Symmetric`/`I` used by later tasks)

@testset "grad_predict matches ForwardDiff on the posterior mean" begin
    Random.seed!(3)
    ℓ = 0.7
    X = [randn(2) for _ in 1:8]; y = randn(8)
    g = update(ExactGP(with_lengthscale(SqExponentialKernel(), ℓ); noise=1e-6), X, y)
    for x in (randn(2), randn(2), [0.3, -0.4])
        μ∇, Σdiag, H = grad_predict(g, x)
        @test μ∇ ≈ ForwardDiff.gradient(u -> predmean(g, u), x)  rtol=1e-6
        @test H  ≈ ForwardDiff.hessian(u -> predmean(g, u), x)   rtol=1e-6
        @test H  ≈ H'                                            atol=1e-10  # symmetric
        @test all(0 .≤ Σdiag .≤ 1/ℓ^2 + 1e-8)                               # valid, prior-reduced
    end
    # far from data → gradient variance approaches the prior 1/ℓ²
    _, Σfar, _ = grad_predict(g, [50.0, 50.0])
    @test all(Σfar .≈ 1/ℓ^2)
end

@testset "grad_predict is kernel-generic (Matérn-5/2, no hand-derived blocks)" begin
    # The package thesis is to bridge KernelFunctions + AD, not hand-code per-kernel algebra.
    # grad_predict must work for a kernel whose derivatives we never wrote out: AD on predmean
    # gives μ∇ and H for ANY kernel; only the prior gradient-variance coincidence term needs a
    # per-family constant (Matérn's ‖·‖ is non-smooth at r=0, so AD NaNs there) — here 5/(3ℓ²).
    Random.seed!(5)
    ℓ = 0.8
    X = [randn(2) for _ in 1:10]; y = randn(10)
    g = update(ExactGP(with_lengthscale(Matern52Kernel(), ℓ); noise=1e-6), X, y)
    for x in (randn(2), [0.2, -0.5])
        μ∇, Σdiag, H = grad_predict(g, x)
        @test μ∇ ≈ ForwardDiff.gradient(u -> predmean(g, u), x)  rtol=1e-6
        @test H  ≈ ForwardDiff.hessian(u -> predmean(g, u), x)   rtol=1e-6
        @test all(isfinite, Σdiag) && all(Σdiag .≥ 0)
    end
    _, Σfar, _ = grad_predict(g, [50.0, 50.0])
    @test all(Σfar .≈ 5/(3ℓ^2))      # Matérn-5/2 analytic prior gradient variance, σ²=1
end

@testset "prior gradient variance scales with the kernel output scale σ²" begin
    # σ_f² must flow through grad_predict (the explore band) — not be hardcoded to unit variance.
    ℓ = 0.6; σ² = 9.0
    k = σ² * with_lengthscale(SqExponentialKernel(), ℓ)
    @test _outputscale(k) ≈ σ²
    g = ExactGP(k; noise=1e-6)
    _, Σprior, _ = grad_predict(g, [0.0, 0.0])   # no data → pure prior
    @test all(Σprior .≈ σ²/ℓ^2)
end
