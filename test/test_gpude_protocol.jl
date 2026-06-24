using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Statistics, Test
using Magpie: CompositeField, ExactGPField, FieldLayout, gpfield, solve_alpha, predmean, train!, posterior
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
    du[1] = v - v^3 / 3 - w + _I
    du[2] = (v + _a - _b * w) / _τ
    return nothing
end

fhn_known(u, t) = [u[1] - u[2] + _I, (u[1] + _a - _b * u[2]) / _τ]
fhn_residual_true(u) = [-u[1]^3 / 3, 0.0]

# ---------------------------------------------------------------------------
# Helper: inline field_error at a set of state points.
# field_error = median over pts of ‖posterior_mean(gps, z) − true_residual(z)‖
# ---------------------------------------------------------------------------
function _field_err(gps, true_residual, pts)
    errs = [
        norm([predmean(gps[i], z) for i in eachindex(gps)] .- true_residual(z))
            for z in pts
    ]
    return Statistics.median(errs)
end

@testset "CompositeField protocol: R3 gradient gate" begin
    # Use the same short-horizon, single-trajectory gradient-gate pattern as test_exemplar_B_grad.jl.
    # This checks that the `known` closure composes through GaussAdjoint+Mooncake without breaking.
    Random.seed!(42)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)

    u0 = [-1.0, 1.0]; tspan = (0.0, 5.0)
    ts = collect(range(tspan...; length = 12))
    target = Array(solve(ODEProblem(fhn_full!, u0, tspan), Tsit5(); saveat = ts))

    Z = [target[:, i] for i in 1:size(target, 2)]  # anchors on trajectory
    inner = ExactGPField(SqExponentialKernel(), Z; d = 2, logℓ0 = log(0.5))
    cf = CompositeField(fhn_known, inner)

    # Build loss via field_loss(CompositeField, SingleShooting, data)
    loss = ext.field_loss(cf, Magpie.SingleShooting(), [(ts, target)])

    # Evaluate at a point with small nonzero w-weights so the GP RHS is live and
    # wnorm is bounded — at w=0 the GP term is zero so the FHN trajectory is governed
    # only by `known`, producing huge gradients (wnorm≈39225) and inflating FD relerr
    # into the 4e-3 range (only 14% margin below the 5e-3 ceiling).
    rng = Random.MersenneTwister(42)
    v_test = copy(cf.v0)
    v_test[3:end] .= 0.01 .* randn(rng, length(v_test) - 2)

    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v_test)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v_test)[1]

    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    wnorm = norm(g_fd[3:end])   # w-block starts at index 3 (same layout as ExactGPField)

    @info "CompositeField R3 gradient gate" relerr wnorm
    @test relerr < 5.0e-3
    @test wnorm > 1.0e-3
end

@testset "CompositeField: propagate(method=PULL()) throws by design" begin
    # Documented intentional contract: PULL needs the combined Jacobian of known+GP_mean,
    # which is not implemented, so it errors rather than silently producing wrong moments.
    # Pathwise is the supported composite propagator. Lock the contract against silent
    # dispatch regressions (e.g. a refactor falling through to a NaN-producing path).
    Random.seed!(11)
    u0 = [-1.0, 1.0]; tspan = (0.0, 4.0)
    ts = collect(range(tspan...; length = 8))
    target = Array(solve(ODEProblem(fhn_full!, u0, tspan), Tsit5(); saveat = ts))
    Z = [target[:, i] for i in 1:size(target, 2)]
    cf = CompositeField(fhn_known, ExactGPField(SqExponentialKernel(), Z; d = 2))
    @test_throws ErrorException Magpie.propagate(cf, u0, tspan; method = Magpie.PULL(), ts = ts)
    # Pathwise still works (shape + finiteness) — the supported path.
    ens = Magpie.propagate(cf, u0, tspan; method = Magpie.Pathwise(n = 16), ts = ts)
    @test size(ens) == (16, 2, length(ts)) && all(isfinite, ens)
end

@testset "CompositeField protocol: posterior returns residual GP" begin
    # Assert (b): posterior(cf, vopt) returns the residual GP (d ExactGPs, length d, predmean callable)
    Random.seed!(7)

    u0 = [-1.0, 1.0]; tspan = (0.0, 5.0)
    ts = collect(range(tspan...; length = 10))
    target = Array(solve(ODEProblem(fhn_full!, u0, tspan), Tsit5(); saveat = ts))

    Z = [target[:, i] for i in 1:size(target, 2)]
    inner = ExactGPField(SqExponentialKernel(), Z; d = 2)
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
    @test abs(m1) < 1.0e-10
    @test abs(m2) < 1.0e-10
end

@testset "train! returns the mutated field (single return value)" begin
    rng = MersenneTwister(4)
    u0 = [1.0]; tspan = (0.0, 2.0); ts = collect(range(tspan...; length = 10))
    X = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = -0.5u[1]), u0, tspan), Tsit5(); saveat = ts))
    Z = [[x] for x in range(0.2, 1.0; length = 4)]
    field = ExactGPField(SqExponentialKernel(), Z; d = 1)
    ret = train!(field, (ts, X); adam_iters = 20, maxiters = 10)
    @test ret === field                                   # returns the same field, not a tuple
    # field.v0 holds the trained vector: the 1-arg posterior (reads field.v0) matches the explicit form.
    @test predmean(posterior(field)[1], [0.5]) ≈ predmean(posterior(field, field.v0)[1], [0.5])
end

@testset "posterior is the only reconstruction name" begin
    @test !isdefined(Magpie, :posterior_gps)
    @test !isdefined(Magpie, :posterior_sparsegps)
    @test :posterior in names(Magpie)
end

@testset "Capability-B export surface is trimmed" begin
    public = (
        :GPField, :ExactGPField, :SVGPField, :CompositeField, :SparseGP,
        :train!, :posterior, :propagate, :SingleShooting, :MultipleShooting,
        :PULL, :Pathwise, :kmeans_anchors,
    )
    internal = (
        :FieldLayout, :gpfield, :solve_alpha, :unpack, :regularizer,
        :svgp_kl, :nLS, :unpack_LS, :L_ZZ_factor, :svgp_moments,
        :DecoupledGPSample, :build_decoupled_sample,
    )
    exported = Set(names(Magpie))
    for n in public
        @test n in exported
    end
    for n in internal
        @test !(n in exported)          # not exported …
        @test isdefined(Magpie, n)      # … but still defined/callable as Magpie.n
    end
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
        du[1] = v - v^3 / 3 - w + _pI
        du[2] = (v + _pa - _pb * w) / _pτ
        nothing
    end

    local_known(u, t) = [u[1] - u[2] + _pI, (u[1] + _pa - _pb * u[2]) / _pτ]
    local_residual(u) = [-u[1]^3 / 3, 0.0]

    u0 = [-1.0, 1.0]; tspan = (0.0, 8.0)
    ts = collect(range(tspan...; length = 20))
    target = Array(solve(ODEProblem(local_fhn!, u0, tspan), Tsit5(); saveat = ts))

    # Anchors on the trajectory (good coverage of visited states)
    Z = [target[:, i] for i in 1:size(target, 2)]
    inner = ExactGPField(SqExponentialKernel(), Z; d = 2, logℓ0 = log(0.5), lognoise = log(1.0e-2))
    cf = CompositeField(local_known, inner)

    loss0 = ext.field_loss(cf, Magpie.SingleShooting(), [(ts, target)])(cf.v0)
    Magpie.train!(cf, (ts, target); tspan, adam_iters = 300, maxiters = 100, λ = 1 / (20 * 2))
    lossT = ext.field_loss(cf, Magpie.SingleShooting(), [(ts, target)])(cf.v0)

    @info "CompositeField train!" loss0 lossT
    @test lossT < loss0   # training reduced the loss

    # (a) Trajectory RMSE: hard gate — solve the ODE at vopt and measure directly.
    # Previously computed as sqrt(lossT/(N*d)) which INCLUDED the regularizer term,
    # i.e. sqrt((data_mse + reg)/(N*d)), not true trajectory RMSE.
    # Fix: integrate at vopt and compute sqrt(mean(abs2, Array(sol) .- target)).
    pf_opt, rhs_opt! = ext.field_rhs(cf, cf.v0)
    sol_opt = solve(
        ODEProblem((du, u, p, t) -> rhs_opt!(du, u, p, t), u0, tspan, pf_opt),
        Tsit5(); saveat = ts
    )
    sol_arr = Array(sol_opt)
    sol_rmse = size(sol_arr) == size(target) ? sqrt(mean(abs2, sol_arr .- target)) : Inf
    @info "CompositeField sol_rmse (true trajectory RMSE)" sol_rmse
    @test sol_rmse < 0.5   # trajectory recovery (conservative threshold)

    # (b) Residual field error at visited states (advisory / loose)
    gps = Magpie.posterior(cf)
    @test length(gps) == 2      # posterior IS the residual GP

    pts = [target[:, i] for i in 1:size(target, 2)]
    ferr = _field_err(gps, local_residual, pts)
    @info "CompositeField residual field_error at visited states" ferr
    # Advisory assertion: single-trajectory identifiability is limited; we log and assert loosely
    @test ferr < 5.0   # loose (real recovery gives ≪ 1; diverged gives > 10)
end

@testset "CompositeField support matrix (propagate axis)" begin
    # Pin the propagate-axis support matrix for CompositeField:
    #   ExactGPField inner: posterior✓  Pathwise✓  PULL✗ (intentional error)
    #   SVGPField inner:    train!✓ (sampled ELBO)  posterior returns SparseGPs  predmean✓
    # Uses a simple 1D scalar ODE so the test runs fast.
    rng = MersenneTwister(7)
    known(u, t) = -0.3 .* u
    u0 = [1.0]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length = 12))
    X = Array(solve(ODEProblem((du, u, p, t) -> (du .= known(u, t); du .+= 0.2), u0, tspan), Tsit5(); saveat = ts))

    # Exact inner: posterior works; Pathwise works; PULL errors clearly.
    Zx = [[x] for x in range(0.3, 1.0; length = 5)]
    cf_ex = CompositeField(known, ExactGPField(SqExponentialKernel(), Zx; d = 1))
    train!(cf_ex, (ts, X); adam_iters = 100, maxiters = 40)
    @test length(posterior(cf_ex)) == 1
    ens = propagate(cf_ex, u0, tspan; method = Pathwise(n = 16), ts = ts)
    @test size(ens, 3) == length(ts)
    @test_throws Exception propagate(cf_ex, u0, tspan; method = PULL(), ts = ts)

    # SVGP inner: train! now routes to sampled ELBO; posterior returns SparseGPs; predmean callable.
    Zs = kmeans_anchors(X, 5; rng = MersenneTwister(2))
    cf_sv = CompositeField(known, SVGPField(SqExponentialKernel(), Zs; dout = 1))
    train!(cf_sv, (ts, X); nsamples = 8, adam_iters = 100, maxiters = 40)
    @test isfinite(predmean(posterior(cf_sv)[1], [0.5]))
end
