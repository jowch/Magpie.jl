using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, Box, grid_points, grad_predict, update

"""
Newton-polish x0 toward a gradient zero of the GP-mean field. Uses Levenberg–Marquardt
damping `(H̄ + λ_damp·I) \\ μ∇` so the step stays well-defined through a near-singular
Hessian (and never flips direction the way a raw gradient-descent fallback would near a
maximum/saddle); reduces to Newton when H̄ is well-conditioned. Returns `(x, μ∇, H̄)` at
the final iterate so callers can classify without recomputing `grad_predict`.
"""
function _newton_polish(g, x0, box; iters::Int = 12, λ_damp::Real = 1.0e-6)
    x = collect(float.(x0)); μ∇, _, H = grad_predict(g, x)
    for _ in 1:iters
        norm(μ∇) < 1.0e-7 && break
        x = clamp.(x .- (Symmetric(H) + λ_damp * I) \ μ∇, box.lb, box.ub)
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
function critical_points(g, box; per_axis::Int = 35, ε_morse::Real = 1.0e-3, res_tol::Real = 1.0e-2)
    d = length(box.lb)
    polished = [_newton_polish(g, x, box) for x in grid_points(box; per_axis = per_axis)]
    conv = filter(p -> norm(p[2]) < res_tol, polished)         # converged to a gradient zero
    uniq = unique(p -> round.(p[1]; digits = 1), conv)           # dedupe by 0.1-bucketed point
    return map(uniq) do (x, _, H)
        λ = eigvals(Symmetric(H))
        kind = any(<(ε_morse), abs.(λ)) ? :unclassified :
            count(<(0), λ) == 0 ? :min :
            count(<(0), λ) == d ? :max : :saddle
        (point = x, kind = kind, λ = λ)
    end
end

@testset "critical_points finds the single min of a quadratic bowl" begin
    Random.seed!(9)
    f(x) = (x[1] - 0.5)^2 + (x[2] + 0.3)^2
    X = [3 .* rand(2) .- 1.5 for _ in 1:40]
    g = update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.7); noise = 1.0e-6), X, f.(X))
    cps = critical_points(g, Box([-1.5, -1.5], [1.5, 1.5]))
    mins = filter(c -> c.kind == :min, cps)
    @test length(mins) == 1
    @test mins[1].point ≈ [0.5, -0.3] atol = 0.1
end

@testset "critical_points recovers the 9 critical points of a multi-well surface" begin
    # f = (x²-1)² + (y²-1)² has 9 critical points: 4 minima (±1,±1), 1 max (0,0),
    # 4 saddles (±1,0),(0,±1). This asserts EXTRACTION quality on a well-sampled GP.
    # The earlier "active beats random for enumeration" framing was retracted as an extraction
    # artifact (see the critical-point design spec); the genuine active-learning advantage is the
    # TARGETED transition_state search exercised in test_saddle.jl, not enumeration.
    f(x) = (x[1]^2 - 1)^2 + (x[2]^2 - 1)^2
    box = Box([-2.0, -2.0], [2.0, 2.0])
    want = vcat(
        [(:min, [a, b]) for a in (-1.0, 1.0) for b in (-1.0, 1.0)],
        [(:max, [0.0, 0.0])],
        [(:saddle, s) for s in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0])]
    )
    recovered(cps) = count(((k, p),) -> any(c -> c.kind == k && isapprox(c.point, p; atol = 0.25), cps), want)

    Random.seed!(1)
    X = [4 .* rand(2) .- 2 for _ in 1:120]
    g = Magpie.fit(update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.6); noise = 1.0e-4), X, f.(X)))
    @test recovered(critical_points(g, box)) ≥ 7      # extraction recovers most of the 9
end
