# Sampled-ELBO heavy CALIBRATION / RECOVERY gates (training through the solver), gated behind
# MAGPIE_TEST_SCIML_SLOW — each test is a minutes-long Mooncake-compiled training. The real
# correctness gates: calibration ≈ nominal on a NONLINEAR system for BOTH shooting modes (spec
# §5.5), and known+SVGP-residual recovery (CompositeField). SingleShooting is gated on a MILD
# 1-D nonlinear field (single-shooting integrates it cleanly); the stiff 2-D Lotka–Volterra is
# reserved for the MultipleShooting gate, where short segments tame the hard sample-field solves
# (the very reason MS exists). GC.gc() between testsets keeps the peak low.
using Test, Magpie, LinearAlgebra, Random
using OrdinaryDiffEq, SciMLSensitivity, KernelFunctions

@testset "SVGP+SingleShooting on mild nonlinear field: calibrated (spec §5.5, SS)" begin
    truef(u) = -0.6 * u[1] + 0.25 * sin(2 * u[1])          # mild, bounded, single-shooting-friendly
    u0 = [1.5]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length = 30))
    clean = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = truef(u)), u0, tspan), Tsit5(); saveat = ts))
    X = clean .+ 0.02 .* randn(MersenneTwister(1), size(clean))
    field = SVGPField(SqExponentialKernel(), kmeans_anchors(X, 8; rng = MersenneTwister(7)); dout = 1)
    train!(field, (ts, X); nsamples = 6, adam_iters = 300, maxiters = 80)
    ens = propagate(field, u0, tspan; method = Pathwise(64), ts = ts)
    μ, Σ = pathwise_moments(ens)
    cov90 = coverage([collect(clean[:, k]) for k in 1:length(ts)], μ, Σ; level = 0.9)
    width = sum(only(s) for s in Σ[2:end]) / (length(Σ) - 1)
    @info "SS mild-nonlinear calibration" cov90 width
    @test abs(cov90 - 0.9) < 0.2          # clean nonlinear problem; near-nominal
    @test all(s -> all(isfinite, s), Σ)
    @test 1.0e-5 < width < 2.0            # finite, non-vacuous, not collapsed
    GC.gc()
end

@testset "SVGP+MultipleShooting calibrated on nonlinear LV (spec §5.5, MS)" begin
    # Stiff 2-D Lotka–Volterra over a long horizon — the regime MS is for. (Verified cov90≈0.80.)
    lv!(du, u, p, t) = (du[1] = 1.5u[1] - u[1] * u[2]; du[2] = u[1] * u[2] - 3u[2]; nothing)
    u0 = [1.0, 1.0]; tspan = (0.0, 6.0); ts = collect(range(tspan...; length = 60))
    clean = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat = ts, abstol = 1.0e-9, reltol = 1.0e-9))
    X = clean .+ 0.02 .* randn(MersenneTwister(3), size(clean))
    field = SVGPField(SqExponentialKernel(), kmeans_anchors(X, 15; rng = MersenneTwister(7)); dout = 2)
    train!(field, (ts, X); shooting = MultipleShooting(nsegments = 8), nsamples = 8, adam_iters = 300, maxiters = 80)
    ens = propagate(field, u0, tspan; method = Pathwise(64), ts = ts)
    μ, Σ = pathwise_moments(ens)
    cov90 = coverage([collect(clean[:, k]) for k in 1:length(ts)], μ, Σ; level = 0.9)
    width = sum(tr(s) for s in Σ[2:end]) / (length(Σ) - 1)
    @info "MS-LV calibration" cov90 width
    @test abs(cov90 - 0.9) < 0.3          # nonlinear + long horizon; near-nominal
    @test all(s -> all(isfinite, s), Σ)
    @test width < 12.0                    # not vacuous
    GC.gc()
end

@testset "Composite(SVGP) trains + recovers residual + Pathwise ensemble" begin
    # 1D scalar ODE: known = -0.5u, residual = +0.3 (constant offset). The residual GP must learn ≈0.3.
    known(u, t) = -0.5 .* u
    f!(du, u, p, t) = (du .= known(u, t); du .+= 0.3; nothing)
    u0 = [1.0]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 40))
    target = Array(solve(ODEProblem(f!, u0, tspan), Tsit5(); saveat = ts))
    X = target .+ 0.02 .* randn(MersenneTwister(11), size(target))
    cf = CompositeField(known, SVGPField(SqExponentialKernel(), kmeans_anchors(X, 6; rng = MersenneTwister(3)); dout = 1))
    ret = train!(cf, (ts, X); nsamples = 8, adam_iters = 300, maxiters = 80)
    @test ret === cf                                  # guard removed; returns the field
    g = posterior(cf)
    @test g isa Magpie.SparseGP                       # posterior returns ONE multi-output SparseGP
    res = predmean(g, [0.5])
    @test isfinite(res)
    @test abs(res - 0.3) < 0.3                         # recovers the +0.3 residual (not 0)
    ens = propagate(cf, u0, tspan; method = Pathwise(64), ts = ts)   # SparseGP composite dispatch
    @test size(ens) == (64, 1, length(ts))
    @test all(isfinite, ens)
    GC.gc()
end
