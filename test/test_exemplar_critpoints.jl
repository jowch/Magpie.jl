using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Statistics: median
using Magpie: ExactGP, GradStraddle, LocalPenalization, ActiveLearner, observe!, run!,
              posterior_gp, queried_points, Box, grid_points, grad_predict, update

"""
Newton-polish x0 toward a gradient zero of the GP-mean field. Uses Levenberg–Marquardt
damping `(H̄ + λ_damp·I) \\ μ∇` so the step stays well-defined through a near-singular
Hessian (and never flips direction the way a raw gradient-descent fallback would near a
maximum/saddle); reduces to Newton when H̄ is well-conditioned. Returns `(x, μ∇, H̄)` at
the final iterate so callers can classify without recomputing `grad_predict`.
"""
function _newton_polish(g, x0, box; iters::Int=12, λ_damp::Real=1e-6)
    x = collect(float.(x0)); μ∇, _, H = grad_predict(g, x)
    for _ in 1:iters
        norm(μ∇) < 1e-7 && break
        x = clamp.(x .- (Symmetric(H) + λ_damp*I) \ μ∇, box.lb, box.ub)
        μ∇, _, H = grad_predict(g, x)
    end
    return (x, μ∇, H)
end

"""
    critical_points(g, box; per_axis=35, ε_morse=1e-3, res_tol=1e-2)

Survey all critical points of the GP-mean field on `box`: **multi-start Newton** from a
regular grid of seeds, keep the converged (`‖μ∇‖ < res_tol`) and deduplicated iterates,
and classify each by Morse index from `eigvals(Symmetric(H̄))` (`0 → min`, `d → max`,
else `saddle`; near-zero eigenvalue → `:unclassified`).

Newton-from-grid (no CI candidate gate): the earlier `|μ∇| ≤ β√Σ∇` filter is brittle
under good resolution — once the active loop resolves the gradient, `Σ∇ → 0` shrinks the
band and rejects grid points merely *near* a zero. Polishing from grid seeds is robust to
that. Classification uses the posterior-**mean** Hessian (a point estimate); swapping in a
`λ_min` confidence interval recovers Inatsu et al. (2020), *Neural Computation* 32(10).
"""
function critical_points(g, box; per_axis::Int=35, ε_morse::Real=1e-3, res_tol::Real=1e-2)
    d = length(box.lb)
    polished = [_newton_polish(g, x, box) for x in grid_points(box; per_axis=per_axis)]
    conv = filter(p -> norm(p[2]) < res_tol, polished)         # converged to a gradient zero
    uniq = unique(p -> round.(p[1]; digits=1), conv)           # dedupe by 0.1-bucketed point
    return map(uniq) do (x, _, H)
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

@testset "Active learning recovers localized critical points where random fails" begin
    # The honest regime where active learning beats uniform random: critical points localized
    # in a SMALL region of a LARGE domain, on a budget too small to cover the domain uniformly.
    # f = (x²-1)² + (y²-1)² has 9 critical points (4 minima (±1,±1), 1 max (0,0), 4 saddles),
    # all inside [-1,1]², searched over the large box [-6,6]² — so [-1,1]² is ~1/36 of the area.
    #
    # • Uniform random spends its budget across the whole box; only a handful of points land in
    #   the informative centre, so the extraction recovers almost nothing.
    # • Plain GradStraddle mode-collapses (resamples one spot — see test_loop.jl).
    # • GradStraddle + LocalPenalization focuses on the centre (the straddle is repelled from the
    #   steep outer walls toward the gradient zeros) AND spreads within it (the penalty), recovering
    #   all 9. Both pieces are needed.
    # Measured (4 seeds, budget 140): active+LP recovery 7–9 (med 9); uniform random 1–3 (med 2).
    #
    # NB the converse boundary (documented, not asserted): on a SMALL domain (e.g. [-3,3]²) random
    # covers fine and the active loop has no advantage — its value is scarce-budget / large-domain.
    f(x) = (x[1]^2 - 1)^2 + (x[2]^2 - 1)^2
    box = Box([-6.0,-6.0], [6.0,6.0])
    want = vcat([(:min, [a,b]) for a in (-1.0,1.0) for b in (-1.0,1.0)],
                [(:max, [0.0,0.0])],
                [(:saddle, s) for s in ([1.0,0.0],[-1.0,0.0],[0.0,1.0],[0.0,-1.0])])
    recovered(cps) = count(((k,p),) -> any(c -> c.kind == k && isapprox(c.point, p; atol=0.25), cps), want)

    Random.seed!(1)
    al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 1.2); noise=1e-4), GradStraddle(β=1.96))
    for x in [12 .* rand(2) .- 6 for _ in 1:20]; observe!(al, x, f(x)); end
    al.acq = LocalPenalization(al.acq, al.Xs; c=0.5)      # diversity over the live history
    run!(al, f; budget=120, over=box, refit_every=10)
    n_active = recovered(critical_points(posterior_gp(al), box))

    n_random = map(101:103) do s                          # uniform-random control, same budget
        Random.seed!(s); X = [12 .* rand(2) .- 6 for _ in 1:140]
        recovered(critical_points(Magpie.fit(update(ExactGP(with_lengthscale(SqExponentialKernel(),1.2); noise=1e-4), X, f.(X))), box))
    end
    @info "localized critical-point survey" active=n_active random=n_random
    @test n_active ≥ 7                       # active+LP recovers most/all of the 9
    @test maximum(n_random) ≤ 4              # random barely resolves the localized centre
    @test n_active ≥ maximum(n_random) + 4   # decisive active-learning advantage
end
