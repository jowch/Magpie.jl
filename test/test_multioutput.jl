using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, update, nlml

# 2-output target; condition a GP and check the representation + conditioning.
f2(x) = [x[1] + x[2], x[1] - x[2]]
mk(d) = ExactGP(with_lengthscale(SqExponentialKernel(), 0.7); noise = 1e-4, d = d)

@testset "multi-output conditioning: n×d weights, shared Cholesky" begin
    Random.seed!(3)
    X = [randn(2) for _ in 1:8]
    Y = [f2(x) for x in X]                         # vector of length-2 observations
    g = update(mk(2), X, Y)
    @test g.d == 2
    @test size(g.α) == (8, 2)                       # weights are n×d (gp-ude ExactGPField layout)
    @test size(g.δ) == (8, 2)
end

@testset "incremental update == from-scratch (d=2)" begin
    Random.seed!(4)
    X = [randn(2) for _ in 1:6]; Y = [f2(x) for x in X]
    g_scratch = update(mk(2), X, Y)
    g_incr = update(update(mk(2), X[1:3], Y[1:3]), X[4:6], Y[4:6])
    xs = [randn(2) for _ in 1:4]
    @test mean(g_scratch, xs) ≈ mean(g_incr, xs) rtol=1e-9
end

@testset "d=1 nlml unchanged (regression guard)" begin
    Random.seed!(5)
    X = [randn(2) for _ in 1:10]; y = [sum(x) for x in X]
    g = update(mk(1), X, y)
    n = length(y)
    expected = 0.5 * dot(g.δ, g.α) + sum(log, diag(g.C.U)) + 0.5n * log(2π)
    @test nlml(g) ≈ expected rtol=1e-12
end

@testset "_obsmatrix validates observation length" begin
    @test_throws ArgumentError Magpie._obsmatrix([[1.0, 2.0], [3.0]], 2)   # ragged
end
