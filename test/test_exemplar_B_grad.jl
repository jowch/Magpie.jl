using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Test
using Magpie: ExactGPField
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences

@testset "B-grad: through-solver gradient matches FD, ∂loss/∂w live" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    truef(u) = -0.5u + sin(u)
    Z = [[x] for x in range(-3, 3; length = 10)]
    u0 = [2.5]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 10))
    target = Array(solve(ODEProblem((u, p, t) -> [truef(u[1])], u0, tspan), Tsit5(); saveat = ts))
    field = ExactGPField(SqExponentialKernel(), Z; d = 1)
    v0 = vcat(log(1.3), 0.0, log(0.1), zeros(10))   # [logℓ, logσ, logσ_obs, vec(w)]; w=0 off-optimum so loss is non-flat in w
    loss = ext.make_loss(field, u0, tspan, ts, target)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    H = Magpie.nhyp(field)
    @info "B-grad" relerr wnorm = norm(g_fd[(H + 1):end]) dσobs = abs(g_fd[H])
    @test relerr < 5.0e-3                          # Spike 1 saw ~1.3e-4
    @test norm(g_fd[(H + 1):end]) > 1.0e-3         # R3: ∂loss/∂w live (w-block starts after the hyper prefix)
    @test abs(g_fd[H]) > 1.0e-3                    # R3 (Phase 2): the σ_obs slot (last hyper) is live
end

# Fast through-solver TRAINING smoke (Phase 6.2): the gradient gate above covers ∂loss; this covers the
# train! optimizer driver (ADAM→LBFGS wiring, _init_vec, v0 store-back) + posterior reconstruction — the
# one product-path piece not otherwise in the DEFAULT suite (full recovery lives in the gated stage1 test).
# Reuses the Mooncake compile already paid above; a handful of tiny 1D solves, so it stays fast.
@testset "B-smoke: train! descends + posterior reconstructs (default-suite product-path smoke)" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    truef(u) = -0.5u + sin(u)
    Z = [[x] for x in range(-3, 3; length = 10)]
    u0 = [2.5]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 10))
    target = Array(solve(ODEProblem((u, p, t) -> [truef(u[1])], u0, tspan), Tsit5(); saveat = ts))
    field = ExactGPField(SqExponentialKernel(), Z; d = 1)
    loss0 = ext.make_loss(field, u0, tspan, ts, target; λ = 0.0, λσ = 0.0)(field.v0)
    Magpie.train!(field, (ts, target); tspan, adam_iters = 30, maxiters = 10)
    lossT = ext.make_loss(field, u0, tspan, ts, target; λ = 0.0, λσ = 0.0)(field.v0)
    @info "B-smoke" loss0 lossT
    @test lossT < loss0                               # train! actually descended (optimizer-driver wiring works)
    gps = Magpie.posterior(field)
    @test length(gps) == 1                            # posterior reconstruction works
    @test isfinite(Magpie.predmean(gps[1], [0.5]))    # reconstructed field is callable
end
