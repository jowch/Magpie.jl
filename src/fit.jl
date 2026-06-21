using Optimization, OptimizationOptimJL, DifferentiationInterface

# Recover ℓ from with_lengthscale(SqExponentialKernel(), ℓ) == SqExp ∘ ScaleTransform(1/ℓ).
_lengthscale(k) = 1 / only(k.transform.s)

@doc raw"""
    nlml(g::ExactGP) -> Real

Negative log marginal likelihood, the objective [`fit`](@ref) minimizes:

```math
-\log p(y \mid X) = \tfrac{1}{2}\,\delta^\top \alpha
    + \sum_i \log C_{ii} + \tfrac{n}{2}\log 2\pi,
```

where ``\delta = y - m(X)``, ``\alpha = C^{-1}\delta``, and ``C`` is the upper Cholesky
factor of ``K + \sigma^2 I``. Returns `0.0` for an unconditioned GP.
"""
function nlml(g::ExactGP)
    _hasdata(g) || return 0.0
    n = length(g.δ)
    return 0.5*dot(g.δ, g.α) + sum(log, diag(g.C.U)) + 0.5n*log(2π)
end

"""
    fit(g::ExactGP; restarts=1, ad=AutoForwardDiff()) -> ExactGP

Optimize the kernel lengthscale by minimizing [`nlml`](@ref) with LBFGS, returning a
GP re-conditioned at the best lengthscale found.

Optimization runs in log-space (`logℓ`, unconstrained, bounded to `[-6, 6]`) so the
recovered `ℓ = exp(logℓ)` stays positive. With `restarts > 1`, extra runs start from
the initial `logℓ` jittered in log-space and the lowest-NLML result wins. `ad` selects
the DifferentiationInterface backend for the gradient.

!!! note
    v1 assumes the prior kernel is `with_lengthscale(SqExponentialKernel(), ℓ)`.
"""
function fit(g::ExactGP; restarts::Int=1, ad=AutoForwardDiff())
    @assert g.prior.kernel isa KernelFunctions.TransformedKernel "v1 fit assumes with_lengthscale(SqExponentialKernel(), ℓ)"
    X = g.x; y = g.δ .+ AbstractGPs.mean(g.prior, g.x)
    noise = g.noise
    logℓ0 = log(_lengthscale(g.prior.kernel))
    # Closure over a length-1 vector [logℓ]; kept AD-compatible (no ParameterHandling unflatten).
    loss(flat, _) = nlml(update(ExactGP(with_lengthscale(SqExponentialKernel(), exp(only(flat))); noise=noise), X, y))
    best = g; best_nlml = nlml(g)
    for r in 1:restarts
        logℓ_start = r == 1 ? [logℓ0] : [logℓ0 + 0.1*randn()]   # first run exact, rest jittered
        prob = OptimizationProblem(OptimizationFunction(loss, ad), logℓ_start; lb=[-6.0], ub=[6.0])
        sol = solve(prob, LBFGS())
        gp_cand = update(ExactGP(with_lengthscale(SqExponentialKernel(), exp(only(sol.u))); noise=noise), X, y)
        if nlml(gp_cand) < best_nlml; best, best_nlml = gp_cand, nlml(gp_cand); end
    end
    return best
end
