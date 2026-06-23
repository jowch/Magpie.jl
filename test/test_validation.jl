using Magpie, KernelFunctions, Test
using Magpie: ExactGP, LaplaceGP, update, Box, grid_points

@testset "observation validation" begin
    g = ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-4)
    @test_throws ArgumentError update(g, [[0.0], [1.0]], [0.0])          # count mismatch
    @test_throws ArgumentError update(g, Vector{Float64}[], Float64[])    # empty
    @test_throws ArgumentError update(g, [[0.0], [NaN]], [0.0, 1.0])      # non-finite input
    @test_throws ArgumentError update(g, [[0.0], [1.0]], [0.0, Inf])      # non-finite value
    @test_throws ArgumentError update(g, [[0.0], [0.0, 1.0]], [0.0, 1.0]) # inconsistent dim
end

@testset "Box validation" begin
    @test_throws ArgumentError Box([0.0, 0.0], [1.0])      # length mismatch
    @test_throws ArgumentError Box([1.0, 0.0], [0.0, 1.0]) # lb > ub at dim 1
end

@testset "grid_points high-D guard" begin
    @test_throws ArgumentError grid_points(Box(fill(-1.0, 5), fill(1.0, 5)); per_axis = 50)
end

@testset "LaplaceGP label coercion" begin
    g = LaplaceGP(with_lengthscale(SqExponentialKernel(), 0.5))
    X = [[0.0], [1.0], [2.0]]
    @test update(g, X, [0, 1, 1]) isa LaplaceGP        # {0,1} integers
    @test update(g, X, [-1, 1, -1]) isa LaplaceGP      # {-1,+1}
    @test update(g, X, [true, false, true]) isa LaplaceGP
    @test_throws ArgumentError update(g, X, [0, 2, 1])  # invalid label set
end
