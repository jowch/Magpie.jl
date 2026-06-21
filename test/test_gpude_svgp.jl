# Pure SVGP math unit tests — NO SciML/OrdinaryDiffEq import.
# Covers: svgp_kl ≥ 0, μ=0,S=I ⇒ KL=0, unpack_LS diagonal = exp(raw),
#         svgp_moments finite, SparseGP mean/var/cov, near-singular K_ZZ robustness.

using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Statistics, Random, Test
using Magpie: svgp_kl, unpack_LS, nLS, svgp_moments, L_ZZ_factor, SparseGP, predmean

@testset "SVGP math: KL ≥ 0, μ=0 S=I ⇒ KL=0, unpack_LS diagonal = exp(raw)" begin
    M = 5

    # μ=0, S=I → KL = 0
    S_I = LowerTriangular(Matrix(1.0I, M, M))
    @test svgp_kl(zeros(M), S_I) ≈ 0.0 atol=1e-12

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
    raw_off  = 0.1 .* randn(rng, nLS(M) - M)
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
    prior  = AbstractGPs.GP(kernel)

    # Inducing points in 1D
    Z = [[z] for z in range(-2.0, 2.0; length=M)]

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
    @test σ2_u ≥ -1e-10  # non-negative within numerical tolerance

    pm = predmean(g, u)
    @test isfinite(pm)

    # mean/var/cov on a vector of inputs
    xs = [[x] for x in range(-1.5, 1.5; length=6)]
    ms = Statistics.mean(g, xs)
    vs = Statistics.var(g, xs)
    @test length(ms) == length(xs)
    @test all(isfinite, ms)
    @test all(isfinite, vs)
    @test all(v -> v ≥ -1e-10, vs)

    # cov(g, xs) — full covariance matrix, should be symmetric (within float tol)
    Σ = Statistics.cov(g, xs)
    @test size(Σ) == (length(xs), length(xs))
    @test all(isfinite, Σ)
    @test norm(Σ - Σ') < 1e-10  # symmetric

    # cov(g, xs, ys) — cross-covariance
    ys = [[x] for x in range(-0.5, 0.5; length=4)]
    C = Statistics.cov(g, xs, ys)
    @test size(C) == (length(xs), length(ys))
    @test all(isfinite, C)
end

@testset "SVGP: near-singular K_ZZ robustness (duplicate inducing points)" begin
    rng = MersenneTwister(99)
    M = 4
    kernel = SqExponentialKernel()
    prior  = AbstractGPs.GP(kernel)

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
