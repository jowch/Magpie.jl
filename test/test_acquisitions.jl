using AlphaGP, AbstractGPs, KernelFunctions, LinearAlgebra, ForwardDiff, Test
using AlphaGP: ExactGP, Straddle, RandStraddle, resample
@testset "Straddle" begin
    g = AlphaGP.update(ExactGP(with_lengthscale(SqExponentialKernel(),0.5); noise=1e-4), [[0.0],[1.0]], [0.0,1.0])
    a = Straddle(h=0.5); x = [0.3]
    # independent oracle: compute μ,σ directly, assert the closed form (not by re-calling a())
    μ = only(mean(g,[x])); σ = sqrt(only(var(g,[x])))
    @test a(g, x) ≈ 1.96*σ - abs(μ - 0.5) rtol=1e-10
    @test all(isfinite, ForwardDiff.gradient(z -> a(g, z), x))
    g0 = AlphaGP.update(ExactGP(with_lengthscale(SqExponentialKernel(),0.5); noise=1e-10), [[0.0]], [0.0])
    @test Straddle(h=0.0)(g0, [0.0]) ≈ 0.0 atol=1e-3          # σ→0 ⇒ -|μ-h| ⇒ 0
end

using Random
@testset "RandStraddle + resample" begin
    @test resample(Straddle(h=0.0)) isa Straddle
    a = RandStraddle(h=0.0, rng=MersenneTwister(0))
    a2 = resample(a); @test a2.sβ != a.sβ                     # fresh draw from the kept rng
    g = AlphaGP.update(ExactGP(with_lengthscale(SqExponentialKernel(),0.5); noise=1e-3), [[0.0],[1.0]], [0.0,1.0])
    x = [0.4]; μ = only(mean(g,[x])); σ = sqrt(only(var(g,[x])))
    @test a(g, x) ≈ max(min(μ + a.sβ*σ - a.h, a.h - (μ - a.sβ*σ)), 0.0) rtol=1e-10
    @test a(g, x) ≥ 0.0                                       # clamp holds
end
