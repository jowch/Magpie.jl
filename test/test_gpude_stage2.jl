using Magpie, KernelFunctions, LinearAlgebra, Random, Statistics, Test
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences
using Magpie: ExactGPField, FieldLayout, MultipleShooting, CompositeField, SVGPField, SingleShooting, kmeans_anchors

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
    # build the flat vector: [field params (nhyp(field) + n*d) ; vec(s0) (d*S)]; nhyp=[logℓ,logσ,logσ_obs(1..d)]
    seg_idx = round.(Int, range(1, length(ts); length = S + 1))
    s0 = hcat([target[:, seg_idx[i]] for i in 1:S]...)
    v0 = vcat(log(0.9), 0.0, fill(log(0.1), d), 0.1 .* randn(n * d), vec(s0))   # [logℓ, logσ, logσ_obs(1..d), vec(w), vec(s0)]
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config = nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    @info "Stage2 grad" relerr = relerr
    @test relerr < 5.0e-3
    off = Magpie.nhyp(field) + n * d                    # s0 block starts after [logℓ,logσ,logσ_obs(1..d),vec(w)]
    s2 = (off + d + 1):(off + 2d)                       # free second node s0[:,2]
    @test norm(g_fd[s2]) > 1.0e-4                        # per-segment node differentiates
end

@testset "Stage 2: continuity + anchor penalties match closed form (zero field isolates them)" begin
    # The gradient gate above is tautological for the penalties: the data-misfit term alone makes
    # s0 differentiable, so it would pass even if λ·cont / λ0·anchor were dropped or mis-weighted.
    # Here we isolate the penalties exactly. With w=0 the GP field is identically zero, so du=0 and
    # each segment's endpoint equals its node (endp_i = s0[:,i]) — no solve dependence. The data term
    # AND the regularizer are then identical between a penalized and an unpenalized loss at the same v,
    # so their difference is EXACTLY λ·Σ‖s0_i−s0_{i+1}‖² + λ0·‖s0_1−X_1‖², a closed form.
    Random.seed!(5)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    Z = [randn(2) for _ in 1:6]; n, d = 6, 2
    u0 = [1.0, 0.5]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length = 7))
    target = Array(solve(ODEProblem((u, p, t) -> [-0.5u[1], -0.4u[2]], u0, tspan), Tsit5(); saveat = ts))
    field = ExactGPField(SqExponentialKernel(), Z; d = d)   # v0 weights are 0 ⇒ gpfield ≡ 0 ⇒ du = 0
    S = 3
    λ, λ0 = 100.0, 1.0e4
    loss_pen = ext.build_loss(field, FieldLayout(n, d), target, ts, tspan, MultipleShooting(nsegments = S; λ = λ, λ0 = λ0))
    loss_nopen = ext.build_loss(field, FieldLayout(n, d), target, ts, tspan, MultipleShooting(nsegments = S; λ = 0.0, λ0 = 0.0))

    s0 = [0.1 0.4 0.9; 0.2 0.5 1.0]   # d×S arbitrary nodes (NOT data-seeded ⇒ both penalties nonzero)
    v = vcat(log(0.9), 0.0, fill(log(0.1), d), zeros(n * d), vec(s0))   # w = 0; logσ_obs is length-d
    cont_expected = sum(sum(abs2, s0[:, i] .- s0[:, i + 1]) for i in 1:(S - 1))
    anchor_expected = sum(abs2, s0[:, 1] .- target[:, 1])
    @test loss_pen(v) - loss_nopen(v) ≈ λ * cont_expected + λ0 * anchor_expected rtol = 1.0e-8

    # All nodes = data IC ⇒ continuity gap 0 AND anchor 0 ⇒ both penalties vanish.
    v_cont = vcat(log(0.9), 0.0, fill(log(0.1), d), zeros(n * d), vec(repeat(target[:, 1], 1, S)))
    @test abs(loss_pen(v_cont) - loss_nopen(v_cont)) < 1.0e-8
end

# NOTE: the empirical single-vs-multiple-shooting RECOVERY contrast ("multiple recovers a long
# horizon where single-shooting stalls") is a by-hand DEMONSTRATION, not a unit test. The multiple-
# shooting *mechanism* is covered here by the gradient gate (relerr ~1e-8, ∂loss/∂s0 live) plus the
# penalty closed-form test above. It was previously a ~10-min test double-gated behind
# MAGPIE_TEST_STAGE2_LH, which meant nothing ran it and it silently rotted — removed. No example
# currently uses MultipleShooting (the Phase-6 examples are all single-shooting), so the recovery
# contrast is not wired into CI; run it by hand if you need the numbers.
# Measured noise-free numbers (Julia 1.12.6, 2026-06-21, seed 20, (0,6) LV ≈1.5 periods):
#   single ≈ 1.34 (stalls),  multiple/8-seg ≈ 0.62 (recovers, 2.2× better).
# The NLL objective (Phase 2) shifted these from the old SSE values (single ≈1.49 / multiple ≈0.39).

@testset "train! with MultipleShooting (ExactGPField) runs end-to-end" begin
    rng = MersenneTwister(3)
    a = -0.3
    u0 = [2.0]; tspan = (0.0, 6.0); ts = collect(range(tspan...; length = 36))
    X = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = a * u[1]), u0, tspan), Tsit5(); saveat = ts))
    X .+= 0.02 .* randn(rng, size(X))
    Z = [collect(c) for c in eachcol(X[:, 1:6])]
    field = ExactGPField(Magpie._kernel(0.0, 0.0), Z; d = 1)
    train!(field, (ts, X); shooting = MultipleShooting(nsegments = 4), adam_iters = 300, maxiters = 80)
    @test all(isfinite, field.v0)
    gps = posterior(field)
    μs, _ = propagate(gps, u0, tspan; method = PULL(), ts = ts)
    rmse = sqrt(mean(sum(abs2, μs[k] .- X[:, k]) for k in 1:length(ts)))
    @info "train!→MS e2e" rmse
    @test rmse < 0.5                                  # MS training recovers the trajectory
end

@testset "Composite(Exact) + MultipleShooting trains (E)" begin
    rng = MersenneTwister(11)
    known(u, t) = -0.5 .* u                       # known linear decay
    f!(du, u, p, t) = (du .= known(u, t); du .+= 0.3; nothing)  # truth: residual = +0.3
    u0 = [1.0]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 16))
    target = Array(solve(ODEProblem(f!, u0, tspan), Tsit5(); saveat = ts))
    X = target .+ 0.02 .* randn(rng, size(target))
    Z = kmeans_anchors(X, 6; rng = MersenneTwister(3))
    cf = CompositeField(known, ExactGPField(SqExponentialKernel(), Z; d = 1))
    ret = train!(
        cf, (ts, X); shooting = MultipleShooting(nsegments = 3),
        adam_iters = 300, maxiters = 80, λ = 1 / 16
    )
    @test ret === cf                                          # returns the field (Task 1 contract)
    res = predmean(posterior(cf)[1], [0.5])                   # reconstructed residual GP mean
    @test isfinite(res)
    @test abs(res - 0.3) < 0.25                               # recovers the +0.3 residual, not 0
end

@testset "Composite(SVGP) + MultipleShooting trains (E)" begin
    # Guard is removed: Composite(SVGP)+MS now routes to svgp_sampled_loss. Verify it runs.
    Zs = [[x] for x in range(-1, 1; length = 5)]
    ts = collect(range(0, 1; length = 8)); X = reshape(collect(range(1.0, 0.5; length = 8)), 1, 8)
    cf_svgp = CompositeField((u, t) -> zero(u), SVGPField(SqExponentialKernel(), Zs; dout = 1))
    ret = train!(cf_svgp, (ts, X); shooting = MultipleShooting(nsegments = 2), nsamples = 4, adam_iters = 20, maxiters = 10)
    @test ret === cf_svgp
    @test all(isfinite, cf_svgp.gp.v0)
end
