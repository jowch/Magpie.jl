using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Test
using Magpie: GPField, ExactGPField, SVGPField, FieldLayout, gpfield, solve_alpha, _kernel, wmat, hyp, kmeans_anchors
using Magpie: unpack, regularizer
using Statistics: var

@testset "gpude unit: field eval + α recompute" begin
    Random.seed!(1)
    Z = [randn(2) for _ in 1:8]; n, d = 8, 2
    field = ExactGPField(SqExponentialKernel(), Z; d=d)
    L = FieldLayout(n, d)
    w = 0.3 .* randn(n, d)
    α = solve_alpha(field, 0.1, 0.0, log(1e-4), w)
    @test size(α) == (n, d)
    # gpfield matches the manual cached-α dot product
    u = randn(2); pf = vcat(0.1, 0.0, vec(α))
    k = _kernel(0.1, 0.0); kuZ = [k(u, z) for z in Z]
    @test gpfield(field, u, pf) ≈ vec(kuZ' * α)
    # k-means returns k anchors in the data convention
    X = randn(2, 50); C = kmeans_anchors(X, 6; rng=MersenneTwister(2))
    @test length(C) == 6 && all(length(c) == 2 for c in C)
end

@testset "GPField protocol: unpack/regularizer (pure, no solver)" begin
    Random.seed!(7)
    # --- ExactGPField ---
    Z = [randn(2) for _ in 1:5]; n, d = 5, 2
    ef = ExactGPField(SqExponentialKernel(), Z; d=d)
    @test ef isa GPField
    pe = unpack(ef, ef.v0)
    @test pe.logℓ == ef.v0[1]
    @test pe.logσ == ef.v0[2]
    @test size(pe.w) == (n, d)
    @test pe.w == wmat(FieldLayout(n, d), ef.v0)        # thin wrapper over existing layout helper
    # at the prior centre (logℓ0=logℓ_ref=0, logσ0=0) the regularizer vanishes
    @test regularizer(ef, ef.v0) ≈ 0 atol=1e-12
    # off-centre logℓ matches the inline ext prior λ(logℓ-ref)²/2s² + λσ logσ²/2sσ²
    v = copy(ef.v0); v[1] = 0.4; v[2] = 0.3
    @test regularizer(ef, v) ≈ 1.0*0.4^2/(2*0.5^2) + 1.0*0.3^2/(2*1.0^2)

    # --- SVGPField ---
    Z0 = [randn(2) for _ in 1:4]
    sf = SVGPField(SqExponentialKernel(), Z0; dout=2)
    @test sf isa GPField
    ps = unpack(sf, sf.v0)
    @test ps.logℓ == sf.v0[1] && ps.logσ == sf.v0[2]
    @test size(ps.Z) == (sf.D, sf.M)
    @test size(ps.μ) == (sf.M, sf.dout)
    @test length(ps.Ls) == sf.dout
    # at v0: μ=0, L_S=I ⇒ KL=0 and logℓ=logℓ_ref ⇒ regularizer ≈ 0
    @test regularizer(sf, sf.v0) ≈ 0 atol=1e-10
end

@testset "decoupled sampler: variance ratio ≈ 1 in-distribution; OOD starvation logged" begin
    Random.seed!(5)
    k = Magpie._kernel(0.0, 0.0)                       # ℓ=1, σ=1
    Z = [[x] for x in range(-2, 2; length=9)]
    gp = Magpie.update(Magpie.ExactGP(k; noise=1e-6), Z, sinpi.(first.(Z)))   # exact-GP oracle (matches the field type)
    drawsamp(sidx) = (uvals = Magpie.mean(gp, Z) .+ Magpie._chol(Magpie.cov(gp, Z) + 1e-8I).L*randn(length(Z));
                      Magpie.build_decoupled_sample(k, Z, uvals; ℓ=1.0, σ=1.0, D=512, rng=MersenneTwister(sidx)))
    ratio_at(xs, S) = (sv = zeros(S, length(xs));
                       for sidx in 1:S; smp = drawsamp(sidx); for (j,x) in enumerate(xs); sv[sidx,j]=smp(x); end; end;
                       vec(var(sv; dims=1)) ./ last(Magpie.mean_and_var(gp, xs)))
    S = 1000
    ratio_in  = ratio_at([[x] for x in range(-1.5, 1.5; length=7)], S)        # in-distribution
    ratio_ood = ratio_at([[4.0], [-4.0]], S)                                  # ≥3 lengthscales past anchors (±2)
    @info "sampler calibration" ratio_in ratio_ood
    @test all(0.85 .< ratio_in .< 1.15)               # in-distribution (band allows S=1000 MC noise)
    # OOD is ADVISORY: a ratio collapsing toward 0 is the variance-starvation signature → bump D. No hard assertion.
    all(ratio_ood .> 0.7) || @info "sampler OOD ratio low — consider larger D (variance starvation)" ratio_ood
end
