using Magpie, AbstractGPs, KernelFunctions, LinearAlgebra, Random, Test
using Magpie: ExactGP, GradStraddle, ActiveLearner, observe!, run!, posterior_gp,
              Box, grid_points, grad_predict, update

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
