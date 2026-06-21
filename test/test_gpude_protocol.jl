using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Statistics, Test
using Magpie: CompositeField, ExactGPField, FieldLayout, gpfield, solve_alpha, predmean
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences

# ---------------------------------------------------------------------------
# Test system: FHN-style UDE split
#   ẋ₁ = x₁ - x₁³/3 - x₂ + I     (full, nonlinear)
#   ẋ₂ = (x₁ + a - b·x₂) / τ
#
# `known(u,t)` = linear part: [u[1] - u[2] + I,  (u[1]+a-b·u[2])/τ]
# true residual = [-u[1]³/3, 0.0]  ← what the GP must learn
# ---------------------------------------------------------------------------

const _a, _b, _τ, _I = 0.7, 0.8, 12.5, 0.5

function fhn_full!(du, u, p, t)
    v, w = u[1], u[2]
    du[1] = v - v^3/3 - w + _I
    du[2] = (v + _a - _b*w) / _τ
    nothing
end

fhn_known(u, t) = [u[1] - u[2] + _I, (u[1] + _a - _b*u[2]) / _τ]
fhn_residual_true(u) = [-u[1]^3/3, 0.0]

# ---------------------------------------------------------------------------
# Helper: inline field_error at a set of state points.
# field_error = median over pts of ‖posterior_mean(gps, z) − true_residual(z)‖
# ---------------------------------------------------------------------------
function _field_err(gps, true_residual, pts)
    errs = [norm([predmean(gps[i], z) for i in eachindex(gps)] .- true_residual(z))
            for z in pts]
    return Statistics.median(errs)
end

@testset "CompositeField protocol: R3 gradient gate" begin
    # Use the same short-horizon, single-trajectory gradient-gate pattern as test_exemplar_B_grad.jl.
    # This checks that the `known` closure composes through GaussAdjoint+Mooncake without breaking.
    Random.seed!(42)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)

    u0 = [-1.0, 1.0]; tspan = (0.0, 5.0)
    ts = collect(range(tspan...; length=12))
    target = Array(solve(ODEProblem(fhn_full!, u0, tspan), Tsit5(); saveat=ts))

    Z = [target[:, i] for i in 1:size(target, 2)]  # anchors on trajectory
    inner = ExactGPField(SqExponentialKernel(), Z; d=2, logℓ0=log(0.5))
    cf = CompositeField(fhn_known, inner)

    # Build loss via field_loss(CompositeField, SingleShooting, data)
    loss = ext.field_loss(cf, Magpie.SingleShooting(), [(ts, target)])

    # Evaluate at a non-zero-weight point so ∂loss/∂w is live
    v0 = copy(cf.v0)

    g_mc = DI.gradient(loss, DI.AutoMooncake(; config=nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]

    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    wnorm  = norm(g_fd[3:end])   # w-block starts at index 3 (same layout as ExactGPField)

    @info "CompositeField R3 gradient gate" relerr wnorm
    @test relerr < 5e-3
    @test wnorm > 1e-3
end

@testset "CompositeField protocol: posterior returns residual GP" begin
    # Assert (b): posterior(cf, vopt) returns the residual GP (d ExactGPs, length d, predmean callable)
    Random.seed!(7)

    u0 = [-1.0, 1.0]; tspan = (0.0, 5.0)
    ts = collect(range(tspan...; length=10))
    target = Array(solve(ODEProblem(fhn_full!, u0, tspan), Tsit5(); saveat=ts))

    Z = [target[:, i] for i in 1:size(target, 2)]
    inner = ExactGPField(SqExponentialKernel(), Z; d=2)
    cf = CompositeField(fhn_known, inner)

    # posterior(cf, v) must return a vector of ExactGPs of length d=2
    gps = Magpie.posterior(cf, cf.v0)
    @test length(gps) == 2
    @test gps[1] isa Magpie.ExactGP
    @test gps[2] isa Magpie.ExactGP

    # predmean callable on both outputs
    test_pt = [0.0, 0.0]
    m1 = predmean(gps[1], test_pt)
    m2 = predmean(gps[2], test_pt)
    @test isfinite(m1) && isfinite(m2)
    @info "CompositeField posterior (prior, w=0)" m1 m2

    # At v0 (w=0), residual GP mean ≈ 0 everywhere (zero weights ⇒ zero posterior mean at prior)
    @test abs(m1) < 1e-10
    @test abs(m2) < 1e-10
end

@testset "CompositeField protocol: train! + trajectory RMSE + residual field_error" begin
    # Full training gate: trajectory RMSE (hard) + residual field_error (advisory / loose).
    # Per identifiability rule: single trajectory does NOT identify the field pointwise in general.
    # We assert sol_rmse < 0.5 (hard) and @info the residual field_error.
    Random.seed!(3)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)

    _pa, _pb, _pτ, _pI = 0.7, 0.8, 12.5, 0.5

    function local_fhn!(du, u, p, t)
        v, w = u[1], u[2]
        du[1] = v - v^3/3 - w + _pI
        du[2] = (v + _pa - _pb*w) / _pτ
        nothing
    end

    local_known(u, t) = [u[1] - u[2] + _pI, (u[1] + _pa - _pb*u[2]) / _pτ]
    local_residual(u) = [-u[1]^3/3, 0.0]

    u0 = [-1.0, 1.0]; tspan = (0.0, 8.0)
    ts = collect(range(tspan...; length=20))
    target = Array(solve(ODEProblem(local_fhn!, u0, tspan), Tsit5(); saveat=ts))

    # Anchors on the trajectory (good coverage of visited states)
    Z = [target[:, i] for i in 1:size(target, 2)]
    inner = ExactGPField(SqExponentialKernel(), Z; d=2, logℓ0=log(0.5), lognoise=log(1e-2))
    cf = CompositeField(local_known, inner)

    loss0 = ext.field_loss(cf, Magpie.SingleShooting(), [(ts, target)])(cf.v0)
    cf, vopt = Magpie.train!(cf, (ts, target); tspan, adam_iters=300, maxiters=100, λ=1/(20*2))
    lossT = ext.field_loss(cf, Magpie.SingleShooting(), [(ts, target)])(vopt)

    @info "CompositeField train!" loss0 lossT
    @test lossT < loss0   # training reduced the loss

    # (a) Trajectory RMSE: hard gate
    sol_rmse = sqrt(lossT / (length(ts) * 2))
    @info "CompositeField sol_rmse" sol_rmse
    @test sol_rmse < 0.5   # trajectory recovery (conservative threshold)

    # (b) Residual field error at visited states (advisory / loose)
    gps = Magpie.posterior(cf, vopt)
    @test length(gps) == 2      # posterior IS the residual GP

    pts = [target[:, i] for i in 1:size(target, 2)]
    ferr = _field_err(gps, local_residual, pts)
    @info "CompositeField residual field_error at visited states" ferr
    # Advisory assertion: single-trajectory identifiability is limited; we log and assert loosely
    @test ferr < 5.0   # loose (real recovery gives ≪ 1; diverged gives > 10)
end
