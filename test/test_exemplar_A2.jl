using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: LaplaceGP, BinaryBALD, ActiveLearner, observe!, run!, posterior_gp, Box, grid_points, predmean

@testset "A2: BinaryBALD + LaplaceGP recovers a decision boundary" begin
    Random.seed!(7)
    label(x) = norm(x) < 1.0
    al = ActiveLearner(LaplaceGP(with_lengthscale(SqExponentialKernel(), 0.6)), BinaryBALD())
    box = Box([-2.0, -2.0], [2.0, 2.0])
    for x in [4 .* rand(2) .- 2 for _ in 1:10]
        observe!(al, x, label(x))
    end
    run!(al, label; budget = 40, over = box)                       # fixed kernel here; LaplaceGP refit is available via refit_every
    g = posterior_gp(al); grid = grid_points(box; per_axis = 40)
    accuracy = mean((predmean(g, p) > 0) == label(p) for p in grid)
    @info "A2 metrics" accuracy
    @test accuracy > 0.9
end
