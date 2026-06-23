using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, update, nlml

# 2-output target; condition a GP and check the representation + conditioning.
f2(x) = [x[1] + x[2], x[1] - x[2]]
mk(d) = ExactGP(with_lengthscale(SqExponentialKernel(), 0.7); noise = 1.0e-4, d = d)

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
    @test mean(g_scratch, xs) ≈ mean(g_incr, xs) rtol = 1.0e-9
end

@testset "d=1 nlml unchanged (regression guard)" begin
    Random.seed!(5)
    X = [randn(2) for _ in 1:10]; y = [sum(x) for x in X]
    g = update(mk(1), X, y)
    n = length(y)
    expected = 0.5 * dot(g.δ, g.α) + sum(log, diag(g.C.U)) + 0.5n * log(2π)
    @test nlml(g) ≈ expected rtol = 1.0e-12
end

@testset "_obsmatrix validates observation length" begin
    @test_throws ArgumentError Magpie._obsmatrix([[1.0, 2.0], [3.0]], 2)   # ragged
end

@testset "multi-output prediction shapes + shared variance" begin
    Random.seed!(6)
    X = [randn(2) for _ in 1:8]; Y = [f2(x) for x in X]
    g2 = update(mk(2), X, Y)
    g1 = update(mk(1), X, [y[1] for y in Y])         # same X, same kernel/noise → same C
    xs = [randn(2) for _ in 1:5]

    M = mean(g2, xs)
    @test size(M) == (5, 2)                           # per-output posterior mean
    @test var(g2, xs) ≈ var(g1, xs) rtol = 1.0e-10        # variance shared across outputs (depends only on X)

    Mv, V = mean_and_var(g2, xs)
    @test Mv ≈ M rtol = 1.0e-10
    @test V ≈ var(g2, xs) rtol = 1.0e-10

    # unconditioned multi-output prior mean is nx×d
    @test size(mean(mk(2), xs)) == (5, 2)
end

@testset "predmean is single-output only" begin
    Random.seed!(7)
    X = [randn(2) for _ in 1:6]
    g2 = update(mk(2), X, [f2(x) for x in X])
    @test_throws ArgumentError Magpie.predmean(g2, randn(2))
    g1 = update(mk(1), X, [sum(x) for x in X])
    @test Magpie.predmean(g1, X[1]) ≈ mean(g1, [X[1]])[1] rtol = 1.0e-12   # d=1 still scalar, matches mean
end

using Magpie: grad_predict, fit, acquire, Box, ActiveLearner, Straddle, observe!

@testset "scalar-only paths reject d>1" begin
    Random.seed!(8)
    X = [randn(2) for _ in 1:6]
    g2 = update(mk(2), X, [f2(x) for x in X])
    @test_throws ArgumentError grad_predict(g2, randn(2))
    @test_throws ArgumentError fit(g2)
    @test_throws ArgumentError acquire(g2, Straddle(); over = Box([-2.0, -2.0], [2.0, 2.0]))
end

@testset "ActiveLearner infers multi-output value storage" begin
    al = ActiveLearner(mk(2), Straddle())
    @test eltype(al.Ys) == Vector{Float64}             # d=2 → vector-valued observations
    al1 = ActiveLearner(mk(1), Straddle())
    @test eltype(al1.Ys) == Float64                    # d=1 unchanged
end
