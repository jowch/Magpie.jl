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

using StatsFuns, AlphaGP
using AlphaGP: LaplaceGP, BinaryBALD
@testset "BinaryBALD Houlsby (logistic)" begin
    Xc = [[x] for x in range(-2,2;length=12)]
    g = AlphaGP.update(LaplaceGP(with_lengthscale(SqExponentialKernel(),1.0)), Xc, first.(Xc) .> 0)
    a = BinaryBALD(); x = [0.3]
    μ, v = mean(g,[x])[1], var(g,[x])[1]
    C = sqrt(π*log(2)/2); λ = sqrt(π/8)
    hb(p) = (q=clamp(p,eps(),1-eps()); -q*log2(q)-(1-q)*log2(1-q))   # bits, matching C
    ref = hb(StatsFuns.normcdf(λ*μ/sqrt(v+1))) - C/sqrt(v+C^2)*exp(-(λ*μ)^2/(2(v+C^2)))
    @test ref > 0                                      # at this near-boundary point the closed form is positive
    @test a(g, x) ≈ ref rtol=1e-8                      # impl's clamp is identity here
    @test a(g, x) ≥ 0
    # high-confidence single-observation point: BALD stays small and ≥ 0
    # (the nats-vs-bits bug produced negative values here; the clamp + bits form fix it)
    g0 = AlphaGP.update(LaplaceGP(with_lengthscale(SqExponentialKernel(),1.0)), [[0.0]], [true])
    @test 0 ≤ a(g0, [0.0]) < 0.5
end
