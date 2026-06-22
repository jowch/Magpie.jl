# Phase 2 (D1): trained observation-noise σ_obs via the Gaussian-NLL data term.
# Env-gated under MAGPIE_TEST_SCIML. Identifiability rule: σ_obs is an observation noise
# (a SCALE, not a ratio) so it IS identifiable and asserted; ℓ/amplitude are NOT asserted.
#
# Mechanism: with the Gaussian NLL `SSE/(2σ²) + (Nd/2)log(2πσ²)`, the σ_obs that minimizes the
# loss tracks the residual RMS. On a NOISY trajectory the irreducible misfit is ≈ the injected
# noise std, so the trained σ_obs should recover σtrue to within a small factor.

using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Test
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using Magpie: ExactGPField, SVGPField, unpack, train!

@testset "trained σ_obs recovers observation noise (ExactGPField)" begin
    Random.seed!(101)
    # 1-D damped-oscillator-ish true field; short horizon so single shooting recovers cleanly.
    truef(u) = -0.5u + sin(u)
    u0 = [2.5]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 15))
    clean = Array(solve(ODEProblem((u, p, t) -> [truef(u[1])], u0, tspan), Tsit5(); saveat = ts))
    σtrue = 0.05
    Xnoisy = clean .+ σtrue .* randn(size(clean))      # X = clean + ε, ε ~ N(0, σ²)

    Z = [[x] for x in range(-1, 3; length = 12)]
    field = ExactGPField(SqExponentialKernel(), Z; d = 1)
    field, vopt = train!(
        field, (ts, Xnoisy); tspan, adam_iters = 800, maxiters = 200,
        λ = 1 / (15), s = 0.5
    )
    σ_obs = exp(unpack(field, vopt).logσ_obs)
    @info "σ_obs recovery (Exact)" σtrue σ_obs ratio = σ_obs / σtrue
    @test 0.5 * σtrue < σ_obs < 2.0 * σtrue
end

@testset "trained σ_obs recovers observation noise (SVGPField)" begin
    Random.seed!(202)
    # 2-output Lotka–Volterra true field (the "unknown" RHS); short horizon, single trajectory.
    lv(u) = [1.5u[1] - u[1] * u[2], u[1] * u[2] - 3u[2]]
    u0 = [1.0, 1.0]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 15))
    clean = Array(solve(ODEProblem((u, p, t) -> lv(u), u0, tspan), Tsit5(); saveat = ts))
    σtrue = 0.05
    Xnoisy = clean .+ σtrue .* randn(size(clean))

    M = 8
    Zg = Magpie.kmeans_anchors(clean, M; rng = MersenneTwister(3))
    field = SVGPField(SqExponentialKernel(), Zg; dout = 2)
    field, vopt = train!(
        field, (ts, Xnoisy); tspan, adam_iters = 800, maxiters = 200,
        λ = 1 / (15 * 2), s = 0.5
    )
    σ_obs = exp(unpack(field, vopt).logσ_obs)
    @info "σ_obs recovery (SVGP)" σtrue σ_obs ratio = σ_obs / σtrue
    @test 0.5 * σtrue < σ_obs < 2.0 * σtrue
end
