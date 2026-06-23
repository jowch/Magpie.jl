# Multi-output SVGP field gradient gate — env-gated under MAGPIE_TEST_SCIML.
# Promotes scratch/spike_capb_svgp_mo.jl (relerr 8.4e-5) to a permanent gate.
# Tests: Mooncake gradient vs finite-differences on the shared-Z multi-output multi-trajectory ELBO;
# asserts ∂/∂Z and ∂/∂μ norms are nonzero (shared-Z + per-output μ both live).

using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Test
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences
using Magpie: SVGPField, nLS

@testset "SVGP-MO: shared-Z multi-output multi-trajectory −ELBO gradient" begin
    Random.seed!(13)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    # 2-output Lotka–Volterra true field (serves as the "unknown" RHS)
    lv(u) = [1.5u[1] - u[1] * u[2], u[1] * u[2] - 3u[2]]
    tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 8))
    ICs = ([1.0, 1.0], [1.3, 0.7])
    trajs = [
        (ts, Array(solve(ODEProblem((u, p, t) -> lv(u), ic, tspan), Tsit5(); saveat = ts)))
            for ic in ICs
    ]
    # 6 shared inducing points in a 2D grid
    M = 6
    Zg = [[x, y] for x in range(0.3, 2.0; length = 3) for y in range(0.3, 2.0; length = 2)]
    field = SVGPField(SqExponentialKernel(), Zg; dout = 2)
    loss = ext.svgp_elbo_loss(field, trajs; tspan)
    # Seed μ ≠ 0 so ∂/∂Z is nonzero (μ=0 ⇒ α=0 ⇒ field≡0 ⇒ vacuous ∂/∂Z, R3).
    # Block offsets route through nhyp(field) (hyper prefix [logℓ,logσ,logσ_obs(1..dout)]); D=2 ⇒ Z is 2·M.
    H = Magpie.nhyp(field)    # = 2 + dout = 4 for SVGPField(dout=2)
    v0 = copy(field.v0)
    v0[(H + 2 * M + 1):(H + 2 * M + M * 2)] .= reduce(vcat, [lv(z) for z in Zg])
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    Zb = (H + 1):(H + 2 * M);  μb = (H + 2 * M + 1):(H + 2 * M + M * 2)
    @info "SVGP-MO grad" relerr norm_dZ = norm(g_fd[Zb]) norm_dμ = norm(g_fd[μb])
    @test relerr < 1.0e-3                                  # spike: 8.4e-5
    # Shared-Z and per-output μ gradients must both be live
    @test norm(g_fd[Zb]) > 1.0e-2
    @test norm(g_fd[μb]) > 1.0e-2
end

@testset "trace-corrected SVGP ELBO: gradient sound + L_S coupled" begin
    rng = MersenneTwister(7)
    Z = [[x] for x in range(-1, 1; length = 3)]
    field = SVGPField(Magpie._kernel(0.0, 0.0), Z; dout = 2)
    ts = collect(range(0.0, 2.0; length = 12))
    Xtrue = hcat([[cos(t), sin(t)] for t in ts]...)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    loss = ext.svgp_elbo_loss(field, [(ts, Xtrue)]; tspan = (0.0, 2.0))
    v = copy(field.v0)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v)[1]
    relerr = norm(g_mc .- g_fd) / (norm(g_fd) + 1.0e-8)
    @test relerr < 1.0e-3
    # An L_S diagonal slot now receives gradient from the data (was ~0 from KL-only at S=I).
    ls_idx = 2 + field.dout + field.D * field.M + field.M * field.dout + 1   # first L_S raw entry
    @test abs(g_mc[ls_idx]) > 1.0e-6
end
