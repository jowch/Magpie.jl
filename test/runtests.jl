using Test
@testset "Magpie" begin
    include("test_spine.jl"); include("test_fit.jl"); include("test_ad.jl")
    include("test_acquisitions.jl"); include("test_laplace.jl")
    include("test_maximize.jl"); include("test_loop.jl"); include("test_exemplar_A1.jl")
    include("test_exemplar_A2.jl")
    include("test_gpude_unit.jl")
    include("test_gpude_svgp.jl")
    include("test_eval.jl")
    include("test_exemplar_B_grad.jl")
    get(ENV, "MAGPIE_TEST_SCIML", "") == "true" && include("test_gpude_stage1.jl")
    get(ENV, "MAGPIE_TEST_SCIML", "") == "true" && include("test_gpude_guard.jl")
    get(ENV, "MAGPIE_TEST_SCIML", "") == "true" && include("test_gpude_protocol.jl")
    # NOTE: the slow LV end-to-end recovery is gated by examples/gp_ude_lotka_volterra.jl's #src
    # assertion (run in CI's docs-examples job), not a standalone test — see the plan's Stage-1 re-plan.
    get(ENV, "MAGPIE_TEST_SCIML", "") == "true" && include("test_gpude_stage2.jl")
    get(ENV, "MAGPIE_TEST_SCIML", "") == "true" && include("test_gpude_svgp_mo.jl")
    get(ENV, "MAGPIE_TEST_SCIML", "") == "true" && include("test_gpude_noise.jl")
    get(ENV, "MAGPIE_TEST_SCIML", "") == "true" && include("test_gpude_pull.jl")
    # Benchmarks live in bench/ and are run by hand (not CI gates) — see bench/timing_exact_vs_svgp.jl.
end
