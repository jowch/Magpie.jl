using Magpie, AbstractGPs, KernelFunctions, Random, Test
using Magpie: ExactGP, LaplaceGP, Straddle, GradStraddle, BinaryBALD, LocalPenalization, ActiveLearner, observe!, run!,
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
    box = Box([-5.0, -5.0], [5.0, 5.0]); cand = Points(grid_points(box; per_axis = 15))  # cheap maximizer
    topcellfrac(pts) = begin
        cells = Dict{Tuple{Int, Int}, Int}()
        for p in pts
            k = (floor(Int, p[1]), floor(Int, p[2])); cells[k] = get(cells, k, 0) + 1
        end
        maximum(values(cells)) / length(pts)
    end
    runloop(divers) = begin
        Random.seed!(7)
        al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 0.8); noise = 1.0e-4), GradStraddle(β = 1.96))
        for x in [10 .* rand(2) .- 5 for _ in 1:15]
            observe!(al, x, f(x))
        end
        divers && (al.acq = LocalPenalization(al.acq, al.Xs; c = 0.6))
        run!(al, f; budget = 40, over = cand, refit_every = 10)
        queried_points(al)[16:end]                      # the acquired (non-seed) points
    end
    plain = topcellfrac(runloop(false))
    div = topcellfrac(runloop(true))
    @test div < plain          # diversity spreads queries vs the plain collapse
    @test div < 0.5            # and no single-cell collapse
end

@testset "acquire throws on input/domain dimension mismatch" begin
    al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-4), Straddle(h = 0.0))
    f1(x) = x[1]^2 - 1
    observe!(al, [-0.5], f1([-0.5]))
    observe!(al, [0.5], f1([0.5]))
    @test_throws ArgumentError acquire(al; over = Box([-1.0, -1.0], [1.0, 1.0]))
end

@testset "ActiveLearner has typed storage and is seedable" begin
    mk() = ActiveLearner(
        ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise = 1.0e-4),
        Magpie.RandStraddle(); rng = MersenneTwister(42)
    )
    al = mk()
    f(x) = sum(x)^2 - 1
    observe!(al, [0.0], f([0.0]))
    @test eltype(al.Xs) != Any                      # concrete input storage
    @test eltype(al.Ys) != Any                      # concrete value storage
    a1 = run!(mk(), f; budget = 8, over = Box([-1.0], [1.0]))
    a2 = run!(mk(), f; budget = 8, over = Box([-1.0], [1.0]))
    @test queried_points(a1) == queried_points(a2)  # same seed → identical survey
end

@testset "ActiveLearner refits a LaplaceGP in the loop (refit_every>0)" begin
    # The loop's refit_every now actually fits the classifier (it was a v1 no-op). Start from a
    # deliberately short lengthscale; after the loop refits, the kernel lengthscale changes — i.e.
    # the classifier hyperparameters were optimized mid-loop, not left untouched.
    Random.seed!(7)
    label(x) = x[1]^2 + x[2]^2 < 1.0
    al = ActiveLearner(LaplaceGP(with_lengthscale(SqExponentialKernel(), 0.2)), BinaryBALD())
    box = Box([-2.0, -2.0], [2.0, 2.0])
    for x in [4 .* rand(2) .- 2 for _ in 1:10]
        observe!(al, x, label(x))
    end
    ℓ0 = Magpie._lengthscale(posterior_gp(al).prior.kernel)
    run!(al, label; budget = 20, over = box, refit_every = 10)   # refits the classifier mid-loop
    ℓ1 = Magpie._lengthscale(posterior_gp(al).prior.kernel)
    @test ℓ1 != ℓ0                                               # fit ran in the loop (no longer a no-op)
    @test posterior_gp(al) isa LaplaceGP
end
