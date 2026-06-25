using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, update, grad_predict, _lengthscale, LocalPenalization, Straddle

@testset "_lengthscale: clear error on ARD/composite, scalar still works" begin
    @test_throws ArgumentError _lengthscale(with_lengthscale(SqExponentialKernel(), [0.5, 1.0]))
    @test_throws ArgumentError _lengthscale(with_lengthscale(SqExponentialKernel(), 0.5) + Matern32Kernel())
    @test _lengthscale(2.0 * with_lengthscale(SqExponentialKernel(), 0.7)) ≈ 0.7
    @test _lengthscale(with_lengthscale(Matern52Kernel(), 1.3)) ≈ 1.3
end

@testset "grad_predict: works for ARD SqExp, clear error for ARD Matern" begin
    Random.seed!(21)
    X = [randn(2) for _ in 1:12]
    y = [sum(abs2, xi) for xi in X]
    g_sq = update(ExactGP(1.0 * with_lengthscale(SqExponentialKernel(), [0.5, 0.8]); noise = 1.0e-4), X, y)
    μ∇, Σ, H = grad_predict(g_sq, randn(2))                 # smooth kernel: AD path, no scalar-ℓ needed
    @test length(μ∇) == 2 && all(isfinite, μ∇) && all(isfinite, Σ)
    g_m = update(ExactGP(1.0 * with_lengthscale(Matern32Kernel(), [0.5, 0.8]); noise = 1.0e-4), X, y)
    @test_throws ArgumentError grad_predict(g_m, randn(2))  # ARD Matern → _prior_grad_var_const → _lengthscale
end

@testset "LocalPenalization: clear error on an ARD kernel" begin
    Random.seed!(22)
    X = [randn(2) for _ in 1:8]
    y = [sum(xi) for xi in X]
    g = update(ExactGP(1.0 * with_lengthscale(SqExponentialKernel(), [0.5, 0.8]); noise = 1.0e-4), X, y)
    lp = LocalPenalization(Straddle(h = 0.0), [randn(2)])   # non-empty pts → reads ℓ
    @test_throws ArgumentError lp(g, randn(2))
end
