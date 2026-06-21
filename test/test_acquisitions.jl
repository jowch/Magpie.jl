using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, ForwardDiff, Test
using Magpie: ExactGP, Straddle, RandStraddle, resample
@testset "Straddle" begin
    g = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-4), [[0.0], [1.0]], [0.0, 1.0])
    a = Straddle(h = 0.5); x = [0.3]
    # independent oracle: compute μ,σ directly, assert the closed form (not by re-calling a())
    μ = only(mean(g, [x])); σ = sqrt(only(var(g, [x])))
    @test a(g, x) ≈ 1.96 * σ - abs(μ - 0.5) rtol = 1.0e-10
    @test all(isfinite, ForwardDiff.gradient(z -> a(g, z), x))
    g0 = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-10), [[0.0]], [0.0])
    @test Straddle(h = 0.0)(g0, [0.0]) ≈ 0.0 atol = 1.0e-3          # σ→0 ⇒ -|μ-h| ⇒ 0
end

using Random
@testset "RandStraddle + resample" begin
    @test resample(Straddle(h = 0.0)) isa Straddle
    a = RandStraddle(h = 0.0, rng = MersenneTwister(0))
    a2 = resample(a); @test a2.sβ != a.sβ                     # fresh draw from the kept rng
    g = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-3), [[0.0], [1.0]], [0.0, 1.0])
    x = [0.4]; μ = only(mean(g, [x])); σ = sqrt(only(var(g, [x])))
    @test a(g, x) ≈ max(min(μ + a.sβ * σ - a.h, a.h - (μ - a.sβ * σ)), 0.0) rtol = 1.0e-10
    @test a(g, x) ≥ 0.0                                       # clamp holds
end

using StatsFuns, Magpie
using Magpie: LaplaceGP, BinaryBALD
@testset "BinaryBALD Houlsby (logistic)" begin
    Xc = [[x] for x in range(-2, 2; length = 12)]
    g = Magpie.update(LaplaceGP(with_lengthscale(SqExponentialKernel(), 1.0)), Xc, first.(Xc) .> 0)
    a = BinaryBALD(); x = [0.3]
    μ, v = mean(g, [x])[1], var(g, [x])[1]
    # INDEPENDENT oracle: true binary BALD (bits) = H_b(E_f σ(f)) − E_f H_b(σ(f)) for the
    # LOGISTIC likelihood, by fine-grid quadrature of N(μ,v). This is a different computation
    # from the impl's Houlsby closed form, so it actually validates calibration (catches the
    # ~2.5× error a closed-form-vs-closed-form check could not).
    σ(f) = 1 / (1 + exp(-f))
    hb(p) = (q = clamp(p, 1.0e-15, 1 - 1.0e-15); -q * log2(q) - (1 - q) * log2(1 - q))
    fs = range(μ - 12 * sqrt(v), μ + 12 * sqrt(v); length = 20001)
    w = exp.(-(fs .- μ) .^ 2 ./ (2v)); w ./= sum(w)
    true_bald = hb(sum(w .* σ.(fs))) - sum(w .* hb.(σ.(fs)))
    @test true_bald > 0
    @test a(g, x) ≈ true_bald rtol = 0.2                 # calibrated to Houlsby approximation error
    @test a(g, x) ≥ 0
    # high-confidence single-observation point: BALD stays small and ≥ 0
    # (the nats-vs-bits bug produced negative values here; the clamp + bits form fix it)
    g0 = Magpie.update(LaplaceGP(with_lengthscale(SqExponentialKernel(), 1.0)), [[0.0]], [true])
    @test 0 ≤ a(g0, [0.0]) < 0.5
end
