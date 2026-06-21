using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Test
using Magpie: ExactGPField, FieldLayout, gpfield, solve_alpha, _kernel, wmat, hyp, kmeans_anchors

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
