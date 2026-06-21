using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Test
using Magpie: ExactGPField, FieldLayout, gpfield, solve_alpha, _kernel, wmat, hyp, kmeans_anchors
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
