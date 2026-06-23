# Pure SVGP math unit tests — NO SciML/OrdinaryDiffEq import.
# Covers: svgp_kl ≥ 0, μ=0,S=I ⇒ KL=0, unpack_LS diagonal = exp(raw),
#         svgp_moments finite, SparseGP mean/var/cov, near-singular K_ZZ robustness.
# Task 4.2: svgp_kl vs direct formula (rtol=1e-10).
# Task 4.1: SVGP = exact GP in the M=N limit (rtol=1e-6 on mean; ~machine eps achieved).

using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Statistics, Random, Test
using Magpie: svgp_kl, unpack_LS, nLS, svgp_moments, svgp_var, L_ZZ_factor, SparseGP, predmean,
    _kernel, ExactGP

@testset "SVGP math: KL ≥ 0, μ=0 S=I ⇒ KL=0, unpack_LS diagonal = exp(raw)" begin
    M = 5

    # μ=0, S=I → KL = 0
    S_I = LowerTriangular(Matrix(1.0I, M, M))
    @test svgp_kl(zeros(M), S_I) ≈ 0.0 atol = 1.0e-12

    # μ≠0, S=I → KL > 0
    @test svgp_kl(ones(M), S_I) > 0

    # General S (non-identity) → KL ≥ 0
    rng = MersenneTwister(42)
    for _ in 1:5
        μ = randn(rng, M)
        # Build a random lower-triangular with positive diagonal
        raw = vcat(0.3 .* randn(rng, M), 0.1 .* randn(rng, nLS(M) - M))
        S_L = unpack_LS(raw, M)
        @test svgp_kl(μ, S_L) ≥ 0.0
    end

    # unpack_LS: diagonal entries = exp(raw diagonal entries)
    raw_diag = 0.5 .* randn(rng, M)
    raw_off = 0.1 .* randn(rng, nLS(M) - M)
    Lraw = vcat(raw_diag, raw_off)  # column-major lower: first M entries are diagonal
    # Actually the column-major layout packs diagonal as j=i in the j,i loop
    # Rebuild properly: indices go j in 1:M, i in j:M  → first entry is (1,1), then (2,1)...
    # Diagonal entries are at positions j, i==j, i.e. at the cumulative starts of each column.
    # Let's just build raw so diag raw = known values and check.
    raw_full = zeros(nLS(M))
    diag_vals = [0.1, 0.5, -0.3, 1.0, -0.7]  # M = 5 diagonal log-values
    idx = 1
    for j in 1:M
        for i in j:M
            if i == j
                raw_full[idx] = diag_vals[j]
            else
                raw_full[idx] = 0.05 * (i + j)  # some off-diagonal
            end
            idx += 1
        end
    end
    LS = unpack_LS(raw_full, M)
    @test all(diag(LS) .≈ exp.(diag_vals))
    @test LS isa LowerTriangular
end

@testset "SVGP: svgp_moments finite + SparseGP predmean/mean/var/cov" begin
    rng = MersenneTwister(7)
    M = 4
    kernel = SqExponentialKernel()
    prior = AbstractGPs.GP(kernel)

    # Inducing points in 1D
    Z = [[z] for z in range(-2.0, 2.0; length = M)]

    # Random variational params
    raw = vcat(zeros(M), 0.2 .* randn(rng, nLS(M) - M))  # diag raw=0 ⇒ exp=1
    L_S = unpack_LS(raw, M)
    μ_v = 0.5 .* randn(rng, M)

    # L_ZZ_factor uses relative jitter — must succeed and be lower-triangular
    L_ZZ = L_ZZ_factor(prior, Z)
    @test L_ZZ isa LowerTriangular
    @test all(isfinite, L_ZZ)

    # SparseGP constructor: α = L_ZZ' \ μ_v
    g = SparseGP(prior, Z, μ_v, L_S)

    # predmean at a single point
    u = [0.3]
    μ_u, σ2_u = svgp_moments(prior, Z, L_ZZ, g.α, L_S, u)
    @test isfinite(μ_u)
    @test isfinite(σ2_u)
    @test σ2_u ≥ -1.0e-10  # non-negative within numerical tolerance

    pm = predmean(g, u)
    @test isfinite(pm)

    # mean/var/cov on a vector of inputs
    xs = [[x] for x in range(-1.5, 1.5; length = 6)]
    ms = Statistics.mean(g, xs)
    vs = Statistics.var(g, xs)
    @test length(ms) == length(xs)
    @test all(isfinite, ms)
    @test all(isfinite, vs)
    @test all(v -> v ≥ -1.0e-10, vs)

    # cov(g, xs) — full covariance matrix, should be symmetric (within float tol)
    Σ = Statistics.cov(g, xs)
    @test size(Σ) == (length(xs), length(xs))
    @test all(isfinite, Σ)
    @test norm(Σ - Σ') < 1.0e-10  # symmetric

    # cov(g, xs, ys) — cross-covariance
    ys = [[x] for x in range(-0.5, 0.5; length = 4)]
    C = Statistics.cov(g, xs, ys)
    @test size(C) == (length(xs), length(ys))
    @test all(isfinite, C)
end

@testset "SVGP: near-singular K_ZZ robustness (duplicate inducing points)" begin
    rng = MersenneTwister(99)
    M = 4
    kernel = SqExponentialKernel()
    prior = AbstractGPs.GP(kernel)

    # Force two nearly-identical inducing points
    Z_dup = [[0.0], [0.0], [1.0], [2.0]]  # first two are identical

    # L_ZZ_factor with relative jitter should still produce a finite result
    L_ZZ = L_ZZ_factor(prior, Z_dup)
    @test all(isfinite, diag(L_ZZ))
    @test all(diag(L_ZZ) .> 0)

    # svgp_kl with those inducing points
    raw = zeros(nLS(M))
    L_S = unpack_LS(raw, M)  # diag = exp(0) = 1, S = I
    μ_v = randn(rng, M)
    kl = svgp_kl(μ_v, L_S)
    @test isfinite(kl)
    @test kl ≥ 0

    # SparseGP predmean should also remain finite
    g = SparseGP(prior, Z_dup, μ_v, L_S)
    u = [0.5]
    pm = predmean(g, u)
    @test isfinite(pm)
end

# ---------------------------------------------------------------------------
# Task 4.2 — svgp_kl vs direct formula (independent KL oracle)
# KL[N(μ,S) ‖ N(0,I)] = 0.5*(tr(S) + μ'μ - M - logdet(S))  with S = L_S L_S'
# The whitened KL in svgp_kl uses logdet(S) = 2*sum(log, diag(L_S)),
# tr(S) = ‖L_S‖²_F, so both forms must agree to rtol=1e-10.
# ---------------------------------------------------------------------------
@testset "Task 4.2: svgp_kl vs direct KL formula (rtol=1e-10)" begin
    rng = MersenneTwister(42)
    M = 5
    for trial in 1:5
        μ = randn(rng, M)
        # Random lower-triangular with positive diagonal
        raw = vcat(0.3 .* randn(rng, M), 0.1 .* randn(rng, nLS(M) - M))
        L_S = unpack_LS(raw, M)
        S = Matrix(L_S) * Matrix(L_S)'
        kl_direct = 0.5 * (tr(S) + dot(μ, μ) - M - logdet(S))
        kl_fn = svgp_kl(μ, L_S)
        @test kl_fn ≈ kl_direct rtol = 1.0e-10
        @test kl_fn ≥ 0.0
    end
end

# ---------------------------------------------------------------------------
# Task 4.1 — SVGP = exact GP in the M=N limit
#
# Construction: set Z = X (inducing = data), noise σ_n² = jitter_rel * s2 so that
# L_ZZ L_ZZ' = K_ZZ + σ_n² I exactly (no approximation from mismatched jitter).
# Whitened mean: μ_v = L_ZZ \ y  ⟹  α = L_ZZ' \ μ_v = (K_ZZ + σ_n² I)⁻¹ y ✓
# Variational covariance: L_S → ε·I (collapsed, near-Dirac) so ‖L_S'A‖² ≈ 0.
# SVGP var ≈ ExactGP var (the ε² term is < 1e-29, negligible).
# Achieved: mean error ≲ 1e-15 (machine eps); var error ≲ 1e-15.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Task 7b — tight oracle for the variational-correction term ‖L_S' A‖²
#
# Uses a NON-trivial L_S so the correction term is nonzero and checked against
# a direct independent computation. The M=N (L_S≈0) oracle never exercises this path.
# Formula: σ² = k(u,u) − dot(A,A) + sum(abs2, L_S'*A),  A = L_ZZ \ k(Z,u)
# ---------------------------------------------------------------------------
@testset "Task 7b: svgp_moments variational-correction term (non-trivial L_S, rtol=1e-10)" begin
    rng = MersenneTwister(17)
    M = 4
    kernel = _kernel(0.0, 0.0)   # ℓ=1, σ=1 (same as Task 4.1)
    prior = AbstractGPs.GP(kernel)

    Z = [[z] for z in range(-1.5, 1.5; length = M)]
    L_ZZ = L_ZZ_factor(prior, Z)   # default jitter=1e-4

    # Non-trivial L_S: diagonal 0.5 + small random off-diagonal entries
    L_S_mat = zeros(M, M)
    for j in 1:M, i in j:M
        L_S_mat[i, j] = (i == j) ? 0.5 : 0.05 * randn(rng)
    end
    L_S = LowerTriangular(L_S_mat)

    # Variational mean (arbitrary α)
    α = randn(rng, M)

    u = [0.4]   # single off-inducing test point

    # Call svgp_moments
    μ_star, σ2 = svgp_moments(prior, Z, L_ZZ, α, L_S, u)

    # Independent reference computation
    kZu = vec(AbstractGPs.cov(prior, Z, [u]))
    A = L_ZZ \ kZu
    correction = sum(abs2, L_S' * A)
    expected_σ2 = only(AbstractGPs.var(prior, [u])) - dot(A, A) + correction

    # Guard: correction must be genuinely nonzero (not another near-zero case)
    @test correction > 0.01

    # Tight self-consistency oracle — catches sign flip, wrong factor (L_S vs L_S'), etc.
    @test σ2 ≈ expected_σ2 rtol = 1.0e-10

    # Mean is unaffected by L_S; sanity-check it too
    expected_μ = only(AbstractGPs.mean(prior, [u])) + dot(kZu, α)
    @test μ_star ≈ expected_μ rtol = 1.0e-10
end

@testset "Task 4.1: SVGP = ExactGP in M=N limit (rtol=1e-6 on mean/var)" begin
    rng = MersenneTwister(1)
    N = 6
    kernel = _kernel(0.0, 0.0)   # ℓ=1, σ=1
    X = [[x] for x in range(-2.0, 2.0; length = N)]
    y = sin.(first.(X))

    # Choose σ_n² that equals the absolute jitter that L_ZZ_factor will add.
    # L_ZZ_factor uses jitter_rel * s2 as the absolute shift.
    # With s2 = mean(diag(K_ZZ)) = 1 for SE kernel at scale σ=1, jitter_rel = σ_n².
    σ_n2 = 1.0e-10
    prior = AbstractGPs.GP(kernel)
    K_ZZ = AbstractGPs.cov(prior, X)
    s2 = mean(diag(K_ZZ))          # should be 1.0
    jitter_rel = σ_n2 / s2             # = 1e-10 when s2=1

    # ExactGP reference with the same noise
    gp_exact = Magpie.update(ExactGP(kernel; noise = σ_n2), X, y)

    # L_ZZ with the matching jitter so L_ZZ L_ZZ' = K_ZZ + σ_n² I exactly
    L_ZZ = L_ZZ_factor(prior, X; jitter = jitter_rel)

    # Whitened variational mean: μ_v = L_ZZ \ y
    # → α_stored = L_ZZ' \ μ_v = (L_ZZ L_ZZ')⁻¹ y = (K_ZZ + σ_n²I)⁻¹ y  ✓
    μ_v = L_ZZ \ y

    # Collapsed variational covariance L_S = ε·I (near-Dirac)
    # → ‖L_S' A‖² = ε² ‖A‖² < 1e-28, negligible vs any variance we test
    ε_S = 1.0e-15
    L_S_mat = zeros(N, N)
    for i in 1:N
        L_S_mat[i, i] = ε_S
    end
    L_S = LowerTriangular(L_S_mat)

    g_svgp = SparseGP(prior, X, μ_v, L_S; jitter = jitter_rel)

    # Off-inducing test points
    xs_test = [[x] for x in [-1.5, -0.7, 0.0, 0.8, 1.3]]

    m_exact = Statistics.mean(gp_exact, xs_test)
    v_exact = Statistics.var(gp_exact, xs_test)
    m_svgp = Statistics.mean(g_svgp, xs_test)
    v_svgp = Statistics.var(g_svgp, xs_test)

    # Mean: should agree to near machine precision (rtol=1e-6 guaranteed; ~1e-15 achieved)
    @test m_svgp ≈ m_exact rtol = 1.0e-6

    # Variance: ε_S² contribution is < 1e-28; exact agreement expected
    @test v_svgp ≈ v_exact rtol = 1.0e-6
end

@testset "svgp_var is AD-safe svgp_moments variance" begin
    Z = [[x] for x in range(-1, 1; length = 4)]
    k = Magpie._kernel(0.0, 0.0)
    prior = AbstractGPs.GP(AbstractGPs.ZeroMean(), k)
    L_ZZ = Magpie.L_ZZ_factor(prior, Z; jitter = 1.0e-4)
    L_S = Magpie.unpack_LS([0.1, 0.0, 0.2, 0.0, 0.0, 0.3, -0.1, 0.0, 0.05, 0.15][1:Magpie.nLS(4)], 4)
    u = [0.3]
    α = L_ZZ' \ zeros(4)
    μ_ref, σ2_ref = Magpie.svgp_moments(prior, Z, L_ZZ, α, L_S, u)
    σ2 = Magpie.svgp_var(prior, Z, L_ZZ, L_S, u)
    @test σ2 ≈ σ2_ref rtol = 1.0e-10           # identical where the clamp is inactive (σ²>0)
    @test σ2 > 0
end
