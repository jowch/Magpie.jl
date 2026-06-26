using Magpie, KernelFunctions, LinearAlgebra, Random, Statistics, Test

# ---------------------------------------------------------------------------
# coverage — calibration on known Gaussian
# ---------------------------------------------------------------------------

@testset "coverage: calibrated on N(0,I)" begin
    Random.seed!(42)
    n_steps = 2000
    d = 3

    # Constant μ=0, Σ=I for all steps
    μs = [zeros(d) for _ in 1:n_steps]
    Σs = [Matrix{Float64}(I, d, d) for _ in 1:n_steps]

    # truth drawn from the exact same distribution
    truth = [randn(d) for _ in 1:n_steps]

    c90 = coverage(truth, μs, Σs; level = 0.9)
    c50 = coverage(truth, μs, Σs; level = 0.5)

    @test abs(c90 - 0.9) < 0.05
    @test abs(c50 - 0.5) < 0.05
end

@testset "coverage tracks level and scale (not tautological)" begin
    rng = MersenneTwister(5)
    d, n = 2, 4000
    truth = [randn(rng, d) for _ in 1:n]
    μs = [zeros(d) for _ in 1:n]
    Σs = [Matrix{Float64}(I, d, d) for _ in 1:n]      # truth ~ N(0,I), μ=0, Σ=I
    c90 = coverage(truth, μs, Σs; level = 0.9)
    c50 = coverage(truth, μs, Σs; level = 0.5)
    @test abs(c90 - 0.9) < 0.03
    @test abs(c50 - 0.5) < 0.03
    @test c90 > c50                                   # higher level ⇒ more coverage (was ignored by the old test)
    # Under-dispersed Σ ⇒ severe under-coverage (scale matters).
    Σtight = [1.0e-3 * Matrix{Float64}(I, d, d) for _ in 1:n]
    @test coverage(truth, μs, Σtight; level = 0.9) < 0.2
end

@testset "coverage: leading Σ=zeros step is excluded, not counted as miss" begin
    # A leading degenerate step (Σ=0, PULL t=0 point mass) must be skipped.
    # The remaining n_valid steps are well-calibrated N(0,I) draws.
    Random.seed!(7)
    d = 3
    n_valid = 2000

    μs_valid = [zeros(d) for _ in 1:n_valid]
    Σs_valid = [Matrix{Float64}(I, d, d) for _ in 1:n_valid]
    truth_valid = [randn(d) for _ in 1:n_valid]

    # With degenerate leading step prepended
    μs_aug = [zeros(d), μs_valid...]
    Σs_aug = [zeros(d, d), Σs_valid...]
    truth_aug = [zeros(d), truth_valid...]

    c_valid = coverage(truth_valid, μs_valid, Σs_valid; level = 0.9)
    c_aug = coverage(truth_aug, μs_aug, Σs_aug; level = 0.9)

    # Both must be well-calibrated (~0.9) and identical (degenerate step excluded)
    @test abs(c_valid - 0.9) < 0.05
    @test c_aug ≈ c_valid

    # Edge case: all-degenerate → NaN
    Σs_zero = [zeros(d, d) for _ in 1:5]
    truth_z = [randn(d) for _ in 1:5]
    μs_zero = [zeros(d) for _ in 1:5]
    @test isnan(coverage(truth_z, μs_zero, Σs_zero; level = 0.9))
end

# ---------------------------------------------------------------------------
# field_error — GP interpolation of a known function
# ---------------------------------------------------------------------------

@testset "field_error: GP recovering a known 1-D function" begin
    Random.seed!(3)
    # True field: u -> [sin(u[1])]
    truefield(u) = [sin(u[1])]

    # Build an ExactGP conditioned on dense anchors; field_error at those anchors → small
    xs_train = [[x] for x in range(0.0, 2π; length = 30)]
    ys_train = [sin(x[1]) for x in xs_train]

    g = ExactGP(SqExponentialKernel(); noise = 1.0e-6)   # single-output (d=1) multi-output GP
    g = update(g, xs_train, ys_train)

    test_pts = [[x] for x in range(0.0, 2π; length = 15)]

    fe = field_error(g, truefield, test_pts)

    @test haskey(NamedTuple(pairs(fe)), :median)
    @test haskey(NamedTuple(pairs(fe)), :q90)
    @test fe.median < 0.05
    @test fe.q90 < 0.1
end

@testset "field_error: median/q90 match StatsBase on a known error set" begin
    # The dense-fit test above leaves median≈q90≈0, so a wrong quantile probability would pass
    # there. Pin the q90 VALUE: a no-data GP predicts 0, so the error at z is ‖truefield(z)‖.
    g = ExactGP(SqExponentialKernel(); noise = 1.0e-6)   # no `update` → mean ≡ 0
    truefield(z) = [z[1]]                                 # error at z = |0 − z[1]| = |z[1]|
    pts = [[x] for x in 1.0:10.0]                          # per-point errors = 1, 2, …, 10
    fe = field_error(g, truefield, pts)
    expected = collect(1.0:10.0)
    @test fe.median ≈ median(expected)                    # = 5.5
    @test fe.q90 ≈ quantile(expected, 0.9)                # matches the exact StatsBase rule the code uses
    @test fe.q90 > fe.median                              # ordering sanity
end

# ---------------------------------------------------------------------------
# recovery_metrics — pure signature, no solver
# ---------------------------------------------------------------------------

@testset "recovery_metrics: pure, no solver" begin
    Random.seed!(5)
    d, n = 2, 50
    # Trivial: pred = truth → traj_rmse = 0
    traj_truth = [randn(d) for _ in 1:n]
    traj_pred = copy(traj_truth)

    truefield(u) = [0.0, 0.0]
    # A trivial GP: one multi-output (d=2) ExactGP with no data — mean returns prior mean (0)
    g = ExactGP(SqExponentialKernel(); noise = 1.0e-6, d = 2)

    offpts = [randn(d) for _ in 1:20]
    rm = recovery_metrics(g, truefield, traj_pred, traj_truth; offpts = offpts)

    @test rm.traj_rmse ≈ 0.0 atol = 1.0e-12
    @test haskey(NamedTuple(pairs(rm)), :field_err_visited)
    @test haskey(NamedTuple(pairs(rm)), :field_err_offmanifold)
    # Both should return (median, q90) NamedTuples
    @test haskey(NamedTuple(pairs(rm.field_err_visited)), :median)
    @test haskey(NamedTuple(pairs(rm.field_err_offmanifold)), :q90)

    # With offpts=nothing, offmanifold is (NaN, NaN)
    rm2 = recovery_metrics(g, truefield, traj_pred, traj_truth)
    @test isnan(rm2.field_err_offmanifold.median)
    @test isnan(rm2.field_err_offmanifold.q90)
end

@testset "traj_rmse: nonzero value matches hand-computed expectation" begin
    # Two steps, each with a known offset vector.
    # Step 1: pred=[1,0], truth=[0,0] → ‖Δ₁‖² = 1
    # Step 2: pred=[0,0], truth=[3,4] → ‖Δ₂‖² = 9+16 = 25
    # RMSE = sqrt(mean(1, 25)) = sqrt(13)
    traj_pred = [[1.0, 0.0], [0.0, 0.0]]
    traj_truth = [[0.0, 0.0], [3.0, 4.0]]
    expected = sqrt(13.0)

    # recovery_metrics needs a GP + truefield; use a trivial multi-output stub (RMSE is pure)
    g = ExactGP(SqExponentialKernel(); noise = 1.0e-6, d = 2)
    truefield(u) = [0.0, 0.0]
    rm = recovery_metrics(g, truefield, traj_pred, traj_truth)

    @test rm.traj_rmse ≈ expected
end

# ---------------------------------------------------------------------------
# ridge_slice — quadratic loss, minimum at known location
# ---------------------------------------------------------------------------

@testset "ridge_slice: minimum at correct grid cell" begin
    c = [0.3, -0.7, 0.0]    # true minimum
    loss(w) = sum(abs2, w .- c)

    xs = range(-2.0, 2.0; length = 41)
    ys = range(-2.0, 2.0; length = 41)

    mat = ridge_slice(loss, c; idx = (1, 2), grid = (xs, ys))

    @test size(mat) == (41, 41)

    # Minimum should be near c[1]=0.3, c[2]=-0.7
    imin = argmin(mat)
    @test abs(xs[imin[1]] - c[1]) < step(xs) + 1.0e-10
    @test abs(ys[imin[2]] - c[2]) < step(ys) + 1.0e-10
end

@testset "ridge_slice: does not mutate v" begin
    v = [1.0, 2.0, 3.0]
    v_copy = copy(v)
    loss(w) = sum(w)
    ridge_slice(loss, v; idx = (1, 2), grid = (range(-1, 1; length = 5), range(-1, 1; length = 5)))
    @test v == v_copy
end

# ---------------------------------------------------------------------------
# pathwise_moments — ensemble → per-step (μ, Σ)
# ---------------------------------------------------------------------------

@testset "pathwise_moments" begin
    # Deterministic ensemble: 3 samples, d=2, T=2.
    ens = zeros(3, 2, 2)
    ens[:, :, 1] = [1.0 0.0; 3.0 0.0; 5.0 0.0]   # step1: x-col mean 3, var 4; y-col 0
    ens[:, :, 2] = [0.0 2.0; 0.0 4.0; 0.0 6.0]   # step2: x 0; y mean 4, var 4
    μs, Σs = pathwise_moments(ens)
    @test length(μs) == 2 && length(Σs) == 2
    @test μs[1] ≈ [3.0, 0.0]
    @test μs[2] ≈ [0.0, 4.0]
    @test Σs[1][1, 1] ≈ 4.0           # sample var of [1,3,5] = 4
    @test size(Σs[1]) == (2, 2)
    @test issymmetric(Σs[2])
    # round-trips into coverage without error
    truth = [μs[k] for k in 1:2]
    @test coverage(truth, μs, Σs .+ Ref(0.1I); level = 0.9) ≥ 0.0
end
