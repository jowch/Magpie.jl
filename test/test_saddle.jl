using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, Box, grid_points, grad_predict, saddle_walk, newton_polish,
    transition_state, classify
using Statistics: mean, std

# --- Müller-Brown potential (standard 2-D test surface with 3 minima, 2 saddles) ---
const MB_A = (-200.0, -100.0, -170.0, 15.0); const MB_a = (-1.0, -1.0, -6.5, 0.7)
const MB_b = (0.0, 0.0, 11.0, 0.6); const MB_c = (-10.0, -10.0, -6.5, 0.7)
const MB_x0 = (1.0, 0.0, -0.5, -1.0); const MB_y0 = (0.0, 0.5, 1.5, 1.0)
function mullerbrown(p)
    x, y = p[1], p[2]; s = 0.0
    for k in 1:4
        dx = x - MB_x0[k]; dy = y - MB_y0[k]
        s += MB_A[k] * exp(MB_a[k] * dx^2 + MB_b[k] * dx * dy + MB_c[k] * dy^2)
    end
    return s
end
const MB_BOX = Box([-1.5, -0.5], [1.0, 2.0]); const CEIL = 200.0
# normalise to ~unit scale so a fixed-ℓ unit-variance kernel is well-conditioned
let g = grid_points(MB_BOX; per_axis = 50), v = min.(mullerbrown.(g), CEIL)
    global const _MBμ = mean(v); global const _MBσ = std(v)
end
mbt(p) = (min(mullerbrown(p), CEIL) - _MBμ) / _MBσ

# minima and the index-1 saddle bracketed by (MB_min, MC)
const MB_min = [-0.05, 0.467]; const MC = [0.623, 0.028]; const S2 = [0.212, 0.293]
const ℓ0 = 0.25; const NOISE = 1.0e-3
mbkernel() = with_lengthscale(SqExponentialKernel(), ℓ0)

# build a GP conditioned on `pts` of the normalised MB surface
buildgp(pts) = Magpie.update(ExactGP(mbkernel(); noise = NOISE), pts, mbt.(pts))

@testset "saddle_walk reaches S2 on a GP conditioned near it" begin
    # a modest sample bracketing S2 (the two basins + the saddle neighbourhood)
    Random.seed!(1)
    pts = [
        MB_min, MC, S2 .+ [0.1, 0.0], S2 .- [0.1, 0.0], S2 .+ [0.0, 0.1], S2 .- [0.0, 0.1],
        (MB_min .+ S2) ./ 2, (MC .+ S2) ./ 2,
    ]
    g = buildgp(pts)
    x, μ∇, H = saddle_walk(g, (MB_min .+ MC) ./ 2; box = MB_BOX)
    @test norm(x .- S2) < 0.1
    @test classify(H) == :saddle
end

@testset "transition_state localizes S2 in a small budget, beating random" begin
    # TARGETED: seed from the two known minima + jittered path, iterate predict/eval/update.
    terrs = Float64[]; rerrs = Float64[]
    for seed in (1, 2, 3)
        Random.seed!(seed)
        res = transition_state(
            mbt, MB_min, MC; kernel = mbkernel(), noise = NOISE,
            box = MB_BOX, budget = 12, nseed = 5, predictor = :minmode
        )
        if seed == 1
            @test norm(res.saddle .- S2) < 0.1
            @test res.kind == :saddle
        end
        push!(terrs, norm(res.saddle .- S2))

        # RANDOM baseline at the SAME budget: 12 uniform pts, extract saddles, nearest to S2.
        Random.seed!(seed)
        X = [MB_BOX.lb .+ (MB_BOX.ub .- MB_BOX.lb) .* rand(2) for _ in 1:12]
        gr = buildgp(X)
        pol = [newton_polish(gr, x; box = MB_BOX, iters = 12) for x in grid_points(MB_BOX; per_axis = 30)]
        conv = filter(p -> norm(p[2]) < 1.0e-2, pol)
        uniq = unique(p -> round.(p[1]; digits = 1), conv)
        sads = [u[1] for u in uniq if classify(u[3]) == :saddle]
        push!(rerrs, isempty(sads) ? Inf : minimum(norm(s .- S2) for s in sads))
    end
    # targeted beats random at this scarce budget (averaged over the fixed seeds)
    @test mean(terrs) < mean(rerrs)
end

@testset "classify is scale-invariant" begin
    Hmin = [1.0 0.0; 0.0 2.0]
    @test classify(Hmin) == :min
    @test classify(1.0e-3 .* Hmin) == :min      # absolute-threshold version returned :unclassified here
    @test classify(1.0e3 .* Hmin) == :min
    Hsaddle = [-1.0 0.0; 0.0 2.0]
    @test classify(1.0e-3 .* Hsaddle) == :saddle
    @test classify([0.0 0.0; 0.0 0.0]) == :unclassified   # genuinely flat → undetermined
    @test classify([1.0e-4 0.0; 0.0 2.0e-4]) == :min   # both eigenvalues < old 1e-3 floor; relative threshold still resolves it
end

@testset "walkers report convergence honestly" begin
    Random.seed!(6)
    # A monotone ramp on the normalised MB box has NO interior critical point: the walker
    # must clamp to a boundary and report converged=false with a non-small residual.
    ramp(p) = p[1]
    X = [MB_BOX.lb .+ (MB_BOX.ub .- MB_BOX.lb) .* rand(2) for _ in 1:20]
    g = Magpie.update(ExactGP(mbkernel(); noise = NOISE), X, ramp.(X))
    r = newton_polish(g, (MB_BOX.lb .+ MB_BOX.ub) ./ 2; box = MB_BOX)
    @test r.converged == false
    @test r.residual > 1.0e-2
    @test r.x == r[1] && r.H == r[3]      # positional compatibility preserved
end

@testset "transition_state is reproducible under a seeded rng" begin
    r1 = transition_state(
        mbt, MB_min, MC; kernel = mbkernel(), noise = NOISE, box = MB_BOX,
        budget = 10, nseed = 5, predictor = :minmode, rng = MersenneTwister(123)
    )
    r2 = transition_state(
        mbt, MB_min, MC; kernel = mbkernel(), noise = NOISE, box = MB_BOX,
        budget = 10, nseed = 5, predictor = :minmode, rng = MersenneTwister(123)
    )
    @test r1.saddle == r2.saddle
end
