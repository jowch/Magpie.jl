using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Statistics: mean, std
using Magpie: ExactGP, GradStraddle, RandGradStraddle, ActiveLearner, observe!, run!,
              posterior_gp, queried_points, Box, grid_points, grad_predict, update

"""
Newton-polish x0 toward a gradient zero of the GP-mean field. Uses Levenberg–Marquardt
damping `(H̄ + λ_damp·I) \\ μ∇` so the step stays well-defined through a near-singular
Hessian (and never flips direction the way a raw gradient-descent fallback would near a
maximum/saddle); reduces to Newton when H̄ is well-conditioned. Returns `(x, μ∇, H̄)` at
the final iterate so callers can classify without recomputing `grad_predict`.
"""
function _newton_polish(g, x0, box; iters::Int=10, λ_damp::Real=1e-6)
    x = collect(float.(x0)); μ∇, _, H = grad_predict(g, x)
    for _ in 1:iters
        norm(μ∇) < 1e-7 && break
        x = clamp.(x .- (Symmetric(H) + λ_damp*I) \ μ∇, box.lb, box.ub)
        μ∇, _, H = grad_predict(g, x)
    end
    return (x, μ∇, H)
end

"""
    critical_points(g, box; per_axis=80, β=1.96, ε_morse=1e-3, res_tol=1e-2)

Survey all critical points of the GP-mean field on `box`: filter grid points whose
gradient CIs all contain 0, Newton-polish each, keep converged & deduplicated points,
and classify by Morse index from `eigvals(Symmetric(H̄))`.

Extension (prior art): swapping the mean-Hessian Morse test here for a `λ_min`
confidence interval — sampling the Hessian posterior — recovers the active-enumeration
method of Inatsu et al. (2020), *Neural Computation* 32(10). See that paper for the
CI-classifier variant; this demo uses the simpler mean-Hessian point estimate.
"""
function critical_points(g, box; per_axis::Int=80, β::Real=1.96, ε_morse::Real=1e-3, res_tol::Real=1e-2)
    d = length(box.lb)
    cands = filter(grid_points(box; per_axis=per_axis)) do x
        μ∇, Σ, _ = grad_predict(g, x; hessian=false)
        all(abs(μ∇[i]) ≤ β*sqrt(Σ[i]) for i in 1:d)
    end
    polished = [_newton_polish(g, x, box) for x in cands]      # each: (x, μ∇, H)
    conv = filter(p -> norm(p[2]) < res_tol, polished)         # reuse μ∇ from polish
    uniq = unique(p -> round.(p[1]; digits=1), conv)           # dedupe by 0.1-bucketed point
    return map(uniq) do (x, _, H)                              # reuse H from polish
        λ = eigvals(Symmetric(H))
        kind = any(<(ε_morse), abs.(λ)) ? :unclassified :
               count(<(0), λ) == 0 ? :min :
               count(<(0), λ) == d ? :max : :saddle
        (point=x, kind=kind, λ=λ)
    end
end

@testset "critical_points finds the single min of a quadratic bowl" begin
    Random.seed!(9)
    f(x) = (x[1]-0.5)^2 + (x[2]+0.3)^2
    X = [3 .* rand(2) .- 1.5 for _ in 1:40]
    g = update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.7); noise=1e-6), X, f.(X))
    cps = critical_points(g, Box([-1.5,-1.5],[1.5,1.5]))
    mins = filter(c -> c.kind == :min, cps)
    @test length(mins) == 1
    @test mins[1].point ≈ [0.5, -0.3] atol=0.1
end

@testset "Active survey recovers all critical points of cos(x)+cos(y)" begin
    # A smooth, single-scale landscape the active learner is well-suited to: exactly 9 critical
    # points with analytic locations — 4 minima (±π,±π), 1 maximum (0,0), 4 saddles (±π,0)/(0,±π).
    Random.seed!(11)
    f(x) = cos(x[1]) + cos(x[2])
    box = Box([-4.0,-4.0], [4.0,4.0])
    al  = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 1.0); noise=1e-4),
                        GradStraddle(β=1.96))
    for x in [8 .* rand(2) .- 4 for _ in 1:20]; observe!(al, x, f(x)); end
    run!(al, f; budget=120, over=box, refit_every=10)   # refit auto-tunes ℓ as data accrues
    cps = critical_points(posterior_gp(al), box; per_axis=80)
    minima  = [[a,b] for a in (-π,π) for b in (-π,π)]
    maxpt   = [0.0, 0.0]
    saddles = [[π,0.0],[-π,0.0],[0.0,π],[0.0,-π]]
    found(kind, p; atol=0.25) = any(c -> c.kind == kind && isapprox(c.point, p; atol=atol), cps)
    @info "cosine survey" n_found=length(cps) kinds=sort(string.([c.kind for c in cps]))
    @test all(found(:min, m) for m in minima)        # all 4 minima, correctly typed
    @test found(:max, maxpt)                          # the maximum
    @test all(found(:saddle, s) for s in saddles)     # all 4 saddles
end

@testset "Himmelblau stress-case: deterministic straddle mode-collapses, randomization spreads" begin
    # Himmelblau is multiscale (sharp quartic minima). The kernel and the extraction handle it
    # GIVEN coverage — uniform random sampling recovers all 9 — so the *acquisition* is the frontier.
    # Controlled finding (coverage = # of the 9 critical-point regions with a sample within 0.6):
    #   det GradStraddle 1/9 · RandGradStraddle ~6/9 (mean over rng seeds) · Inatsu CI-eviction 7/9
    #   · uniform random 9/9.  (Single randomized runs vary ~5–8; det and random are reproducible.)
    # Deterministic GradStraddle mode-collapses (the constant −|μ∇| term pins the argmax to one
    # region on steep walls); randomizing the band breaks the fixation. We assert that ordering.
    # Closing the residual gap to uniform random is open acquisition work (see docs).
    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    box = Box([-5.0,-5.0], [5.0,5.0])
    truth = [[3.0,2.0],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126],  # minima
             [-0.270845,-0.923039],                                                       # maximum
             [3.385154,0.073852],[-3.073026,-0.081353],[-0.127961,-1.953715],[0.086678,2.884255]]  # saddles
    coverage(samples) = count(t -> any(s -> norm(s - t) < 0.6, samples), truth)
    # log1p standardization compresses Himmelblau's 0..800 range onto the unit-variance prior
    Random.seed!(0); ref = log1p.(f.([10 .* rand(2) .- 5 for _ in 1:500]))
    μy, σy = mean(ref), std(ref); g̃(x) = (log1p(f(x)) - μy) / σy
    runcov(acq) = begin
        Random.seed!(11)
        al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 0.8); noise=1e-4), acq)
        for x in [10 .* rand(2) .- 5 for _ in 1:30]; observe!(al, x, g̃(x)); end
        run!(al, g̃; budget=150, over=box, refit_every=10)
        coverage(queried_points(al))
    end
    cov_det  = runcov(GradStraddle(β=1.96))
    cov_rand = runcov(RandGradStraddle(rng=MersenneTwister(1)))   # representative seed (typical ~5–6)
    Random.seed!(11); cov_rng = coverage([10 .* rand(2) .- 5 for _ in 1:200])   # uniform-random control
    @info "Himmelblau acquisition coverage (of 9 regions)" det=cov_det rand=cov_rand uniform_random=cov_rng
    @test cov_det ≤ 2                  # deterministic straddle mode-collapses
    @test cov_rand ≥ cov_det + 2       # randomizing the band clearly spreads coverage
    @test cov_rng ≥ 8                  # kernel & extraction are fine — uniform coverage finds nearly all
end
