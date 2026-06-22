using Magpie, KernelFunctions, LinearAlgebra, Random, Test
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences
using Magpie: ExactGPField, FieldLayout, MultipleShooting

@testset "Stage 2: multiple-shooting gradient matches FD, ∂loss/∂s0 live" begin
    Random.seed!(3)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    truef(u) = [-0.5u[1] + 0.3sin(u[2]), -0.4u[2] + 0.2u[1]]
    Z = [randn(2) for _ in 1:8]; n, d = 8, 2
    u0 = [1.0, 0.5]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length = 7))
    target = Array(solve(ODEProblem((u, p, t) -> truef(u), u0, tspan), Tsit5(); saveat = ts))
    field = ExactGPField(SqExponentialKernel(), Z; d = d)
    S = 3
    ms = MultipleShooting(nsegments = S)
    loss = ext.build_loss(field, FieldLayout(n, d), target, ts, tspan, ms)
    # build the flat vector: [field params (NHYP + n*d) ; vec(s0) (d*S)]; NHYP=[logℓ,logσ,logσ_obs]
    seg_idx = round.(Int, range(1, length(ts); length = S + 1))
    s0 = hcat([target[:, seg_idx[i]] for i in 1:S]...)
    v0 = vcat(log(0.9), 0.0, log(0.1), 0.1 .* randn(n * d), vec(s0))   # [logℓ, logσ, logσ_obs, vec(w), vec(s0)]
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    @info "Stage2 grad" relerr = relerr
    @test relerr < 5.0e-3
    off = Magpie.NHYP + n * d                           # s0 block starts after [logℓ,logσ,logσ_obs,vec(w)]
    s2 = (off + d + 1):(off + 2d)                       # free second node s0[:,2]
    @test norm(g_fd[s2]) > 1.0e-4                        # per-segment node differentiates
end

# NOTE: the empirical single-vs-multiple-shooting RECOVERY contrast ("multiple recovers a long
# horizon where single-shooting stalls") is a DEMONSTRATION, not a unit test — the multiple-shooting
# *mechanism* is already covered by the gradient gate above (relerr ~1e-8, ∂loss/∂s0 live). It was
# previously a ~10-min test double-gated behind MAGPIE_TEST_STAGE2_LH, which meant nothing ran it and
# it silently rotted. Removed. The contrast is shown in a Phase-6 example on NOISY data (the NLL data
# term's intended regime; on noise-free data NLL is a mismatched objective).
# Measured noise-free numbers (Julia 1.12.6, 2026-06-21, seed 20, (0,6) LV ≈1.5 periods):
#   single ≈ 1.34 (stalls),  multiple/8-seg ≈ 0.62 (recovers, 2.2× better).
# The NLL objective (Phase 2) shifted these from the old SSE values (single ≈1.49 / multiple ≈0.39).
