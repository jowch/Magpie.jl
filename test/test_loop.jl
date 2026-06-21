using Magpie, AbstractGPs, KernelFunctions, Random, Test
using Magpie: ExactGP, Straddle, GradStraddle, LocalPenalization, ActiveLearner, observe!, run!,
              acquire, posterior_gp, all_data, queried_points, Box, Points, grid_points

@testset "ActiveLearner loop == batch" begin
    al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-4), Straddle(h = 0.0))
    f(x) = sum(x)^2 - 1
    X = [[x] for x in range(-1, 1; length = 6)]
    for x in X
        observe!(al, x, f(x))
    end
    g_batch = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-4), X, f.(X))
    @test mean(posterior_gp(al), [[0.3], [-0.4]]) ≈ mean(g_batch, [[0.3], [-0.4]]) rtol = 1.0e-9
    @test length(first(all_data(al))) == 6
    @test acquire(al; over = Box([-1.0], [1.0])) isa Vector
    @test_throws ErrorException acquire(al; over = Box([-1.0], [1.0]), q = 3)
end

@testset "LocalPenalization reads live history and prevents query collapse" begin
    # Wrapping the acquisition with LocalPenalization over the learner's LIVE Xs makes the
    # diversity penalty see every appended observation with no loop changes. On Himmelblau,
    # plain GradStraddle mode-collapses (most queries pile into one cell); the penalty spreads them.
    Random.seed!(7)
    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    box  = Box([-5.0,-5.0], [5.0,5.0]); cand = Points(grid_points(box; per_axis=15))  # cheap maximizer
    topcellfrac(pts) = begin
        cells = Dict{Tuple{Int,Int},Int}()
        for p in pts; k = (floor(Int,p[1]), floor(Int,p[2])); cells[k] = get(cells,k,0)+1; end
        maximum(values(cells)) / length(pts)
    end
    runloop(divers) = begin
        Random.seed!(7)
        al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(),0.8); noise=1e-4), GradStraddle(β=1.96))
        for x in [10 .* rand(2) .- 5 for _ in 1:15]; observe!(al, x, f(x)); end
        divers && (al.acq = LocalPenalization(al.acq, al.Xs; c=0.6))
        run!(al, f; budget=40, over=cand, refit_every=10)
        queried_points(al)[16:end]                      # the acquired (non-seed) points
    end
    plain = topcellfrac(runloop(false))
    div   = topcellfrac(runloop(true))
    @test div < plain          # diversity spreads queries vs the plain collapse
    @test div < 0.5            # and no single-cell collapse
end
