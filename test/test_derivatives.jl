# test/test_derivatives.jl
using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test, ForwardDiff
using Magpie: ExactGP, grad_predict, predmean, _lengthscale, update
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
