using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using StatsBase: mean_and_var
using Magpie: ExactGP, Straddle, ActiveLearner, observe!, run!, posterior_gp,
               queried_points, all_data, Box, grid_points, predmean, update, _lengthscale

@testset "A1: Straddle recovers the unit circle level set" begin
    Random.seed!(42)
    f(x) = norm(x) - 1.0
    al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise=1e-4), Straddle(h=0.0))
    box = Box([-2.0,-2.0], [2.0,2.0])
    for x in [4 .* rand(2) .- 2 for _ in 1:10]; observe!(al, x, f(x)); end   # cold start over the FULL box
    run!(al, f; budget=40, over=box, refit_every=10)
    g = posterior_gp(al); grid = grid_points(box; per_axis=50)
    recovery  = mean(sign(predmean(g, p)) == sign(f(p)) for p in grid)
    concentr  = mean(abs(f(p)) < 0.2 for p in queried_points(al))
    @info "A1 metrics" recovery concentr
    @test recovery > 0.95     # boundary recovered
    @test concentr > 0.5      # queries near boundary
    Xall, Yall = all_data(al)                                                # loop == batch invariant
    Yall_f = Float64.(Yall)
    gb = update(ExactGP(with_lengthscale(SqExponentialKernel(), _lengthscale(g.prior.kernel)); noise=1e-4), Xall, Yall_f)
    @test mean(g, grid[1:20]) ≈ mean(gb, grid[1:20]) rtol=1e-6
end
