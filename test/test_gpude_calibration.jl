using Test, Magpie, Statistics, Random
using OrdinaryDiffEq, SciMLSensitivity

@testset "SVGP calibration: trace correction → calibrated + sharp" begin
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

    # Train two SVGP fields from the SAME init: trace ON vs OFF.
    field_on = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 1)
    field_off = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 1)
    train!(field_on, (ts, X); adam_iters = 600, maxiters = 150, trace = true)
    train!(field_off, (ts, X); adam_iters = 600, maxiters = 150, trace = false)

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
                    )
                ),
            )
    ]

    ens_on = propagate(field_on, u0h, (0.0, 5.0); method = Pathwise(256), ts = tsh)
    ens_off = propagate(field_off, u0h, (0.0, 5.0); method = Pathwise(256), ts = tsh)
    μon, Σon = pathwise_moments(ens_on)
    μoff, Σoff = pathwise_moments(ens_off)

    cov_on = coverage(truth, μon, Σon; level = 0.9)
    cov_off = coverage(truth, μoff, Σoff; level = 0.9)
    width_on = mean(only(Σ) for Σ in Σon[2:end])   # mean predictive variance (skip t=0)
    width_off = mean(only(Σ) for Σ in Σoff[2:end])

    @info "SVGP calibration" cov_on cov_off width_on width_off
    # Trace-OFF over-covers with vacuously wide intervals; trace-ON is calibrated AND sharper.
    @test cov_off > 0.97                       # regularized-only ≈ prior variance ⇒ over-covers
    @test abs(cov_on - 0.9) < 0.15            # trace-ON is near-nominal (calibrated)
    @test width_on < 0.6 * width_off          # trace-ON is materially sharper (the real property)
end
