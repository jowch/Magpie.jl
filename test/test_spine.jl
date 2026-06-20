using AlphaGP, AbstractGPs, KernelFunctions, LinearAlgebra, Test
using AlphaGP: ExactGP
@testset "ExactGP contract" begin
    k = with_lengthscale(SqExponentialKernel(), 1.0)
    g0 = ExactGP(k; noise=1e-6)
    @test mean(g0, [0.3, 0.7]) ≈ zeros(2) atol=1e-12         # empty == prior mean
    @test var(g0, [0.3]) ≈ [1.0] atol=1e-8

    x = [0.0, 1.0]; y = [1.0, -1.0]                           # noiseless n=2 hand reference
    K = [k(xi,xj) for xi in x, xj in x]; α_ref = K \ y
    g = AlphaGP.update(ExactGP(k; noise=0.0), x, y)
    @test mean(g, x) ≈ y atol=1e-8
    @test mean(g, [0.5]) ≈ [dot([k(0.5,0.0), k(0.5,1.0)], α_ref)] rtol=1e-10
    @test var(g, [0.0])[1] ≈ 0.0 atol=1e-8
    @test cov(g, [0.5]) isa AbstractMatrix                    # self-cov method exists
end

@testset "incremental == batch" begin
    k = with_lengthscale(SqExponentialKernel(), 0.7); σ² = 1e-3
    X = [randn(2) for _ in 1:8]; y = randn(8); Xt = [randn(2) for _ in 1:5]
    g_batch = AlphaGP.update(ExactGP(k; noise=σ²), X, y)
    g_inc = ExactGP(k; noise=σ²)
    for i in 1:8; g_inc = AlphaGP.update(g_inc, X[i], y[i]); end
    @test mean(g_inc, Xt) ≈ mean(g_batch, Xt) rtol=1e-9
    @test var(g_inc, Xt)  ≈ var(g_batch, Xt)  rtol=1e-9
end
