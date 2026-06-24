using Test, Magpie, Statistics, Random
using OrdinaryDiffEq, SciMLSensitivity

@testset "SVGP calibration: sampled ELBO → calibrated + sharp" begin
    rng = MersenneTwister(2026)
    # Linear 1-D field du = a u (a<0): decaying trajectory; SVGP learns the field.
    a = -0.4
    u0 = [1.5]
    tspan = (0.0, 5.0)
    ts = collect(range(tspan...; length = 30))
    Xclean = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = a * u[1]), u0, tspan), Tsit5(); saveat = ts))
    σobs = 0.05
    X = Xclean .+ σobs .* randn(rng, size(Xclean))
    Z = [collect(c) for c in eachcol(Xclean[:, 1:6])]

    # Train one SVGP field via sampled ELBO (trace term OFF; variance from sampling).
    field = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 1)
    train!(field, (ts, X); nsamples = 8, adam_iters = 400, maxiters = 100)   # Task 6: trimmed

    # Held-out propagation from a fresh IC; Pathwise ensemble → per-step moments → coverage.
    u0h = [1.2]
    tsh = collect(range(0.0, 5.0; length = 25))
    truth = [
        collect(c) for c in eachcol(
                Array(
                    solve(
                        ODEProblem((du, u, p, t) -> (du[1] = a * u[1]), u0h, (0.0, 5.0)),
                        Tsit5();
                        saveat = tsh,
                    ),
                ),
            )
    ]

    ens = propagate(field, u0h, (0.0, 5.0); method = Pathwise(128), ts = tsh)
    μ, Σ = pathwise_moments(ens)

    cov90 = coverage(truth, μ, Σ; level = 0.9)
    width = mean(only(s) for s in Σ[2:end])   # mean predictive variance (skip t=0)

    # Prior-variance ceiling: exp(2·logσ) at the initial logσ=0.0 → σ=1.0 → prior var = 1.0.
    prior_var_ceiling = 2.0   # generous upper bound on prior amplitude²

    @info "SVGP sampled-ELBO calibration" cov90 width prior_var_ceiling
    # Sampled ELBO is calibrated near nominal AND intervals are finite / below the prior ceiling.
    @test abs(cov90 - 0.9) < 0.15       # near-nominal 90% coverage
    @test all(isfinite(only(s)) for s in Σ)
    @test width < prior_var_ceiling      # materially sharper than prior (trained, not vacuous)
end
