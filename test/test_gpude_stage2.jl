using Magpie, KernelFunctions, LinearAlgebra, Random, Test
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences
using Magpie: ExactGPField, FieldLayout, MultipleShooting

@testset "Stage 2: multiple-shooting gradient matches FD, ∂loss/∂s0 live" begin
    Random.seed!(3)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    truef(u) = [-0.5u[1]+0.3sin(u[2]), -0.4u[2]+0.2u[1]]
    Z = [randn(2) for _ in 1:8]; n,d = 8,2
    u0=[1.0,0.5]; tspan=(0.0,3.0); ts=collect(range(tspan...; length=7))
    target = Array(solve(ODEProblem((u,p,t)->truef(u),u0,tspan),Tsit5();saveat=ts))
    field = ExactGPField(SqExponentialKernel(), Z; d=d)
    S = 3
    ms = MultipleShooting(nsegments=S)
    loss = ext.build_loss(field, FieldLayout(n,d), target, ts, tspan, ms)
    # build the flat vector: [field params (3 + n*d) ; vec(s0) (d*S)]
    seg_idx = round.(Int, range(1, length(ts); length=S+1))
    s0 = hcat([target[:, seg_idx[i]] for i in 1:S]...)
    v0 = vcat(log(0.9), 0.0, 0.1 .* randn(n*d), vec(s0))   # [logℓ, logσ, vec(w), vec(s0)]
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config=nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5,1), loss, v0)[1]
    relerr = norm(g_mc.-g_fd)/max(norm(g_fd),eps())
    @info "Stage2 grad" relerr=relerr
    @test relerr < 5e-3
    s2 = (2 + n*d) + d + 1 : (2 + n*d) + 2d          # free second node s0[:,2] (2-prefix layout)
    @test norm(g_fd[s2]) > 1e-4                        # per-segment node differentiates
end

# ---------------------------------------------------------------------------
# Long-horizon contrast test — written but NOT run in the gradient gate job.
# The controller validates this recovery bound separately (slow ADAM recipe, ~20-40 min).
# ---------------------------------------------------------------------------
if get(ENV, "MAGPIE_TEST_STAGE2_LH", "") == "true"
    @testset "Stage 2: multiple shooting trains a long-horizon LV where single-shooting struggles" begin
        Random.seed!(20)
        lv!(du,u,p,t) = (du[1]=1.5u[1]-u[1]*u[2]; du[2]=u[1]*u[2]-3u[2]; nothing)
        u0=[1.0,1.0]; tspan=(0.0,18.0); ts=collect(range(tspan...; length=40))   # long horizon
        target = Array(solve(ODEProblem(lv!,u0,tspan),Tsit5();saveat=ts))
        Z = Magpie.kmeans_anchors(target, 16; rng=MersenneTwister(5))
        ext = Base.get_extension(Magpie, :MagpieSciMLExt)
        L  = Magpie.FieldLayout(16, 2)
        rmse(field, vopt) = sqrt(ext.make_loss(field, L, u0, tspan, ts, target; λ=0.0, λσ=0.0)(vopt) / (40*2))

        # single shooting (baseline, advisory — not asserted; degradation is run/seed-dependent)
        f1 = Magpie.ExactGPField(SqExponentialKernel(), Z; d=2)
        f1, v1 = Magpie.train!(f1, (ts, target); tspan, maxiters=400, λ=1/(40*2))
        @info "Stage-2 contrast (single)" single=rmse(f1, v1)

        # multiple shooting: ~12 segments suffices for this 18-time-unit horizon
        # controller validates this recovery bound
        f2 = Magpie.ExactGPField(SqExponentialKernel(), Z; d=2)
        f2, v2 = Magpie.train!(f2, (ts, target); tspan, maxiters=400, λ=1/(40*2),
                               shooting=Magpie.MultipleShooting(nsegments=12))
        @info "Stage-2 contrast (multiple)" multiple=rmse(f2, v2)
        @test rmse(f2, v2) < 0.5      # multiple shooting trains the long horizon (hard gate)
        # NOTE: single-shooting rmse is logged, not asserted — its degradation is run/seed-dependent.
    end
end
