using Optimization, OptimizationOptimJL, DifferentiationInterface

# Peel ScaledKernel/TransformedKernel wrappers to read hyperparameters.
# ℓ from with_lengthscale(k, ℓ) == k ∘ ScaleTransform(1/ℓ); σ² from `c * k` == ScaledKernel.
_lengthscale(k) = 1 / only(k.transform.s)
_lengthscale(k::KernelFunctions.ScaledKernel) = _lengthscale(k.kernel)
_outputscale(k) = 1.0
_outputscale(k::KernelFunctions.ScaledKernel) = only(k.σ²) * _outputscale(k.kernel)
_basekernel(k) = k
_basekernel(k::KernelFunctions.TransformedKernel) = _basekernel(k.kernel)
_basekernel(k::KernelFunctions.ScaledKernel) = _basekernel(k.kernel)

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
    return 0.5 * dot(g.δ, g.α) + sum(log, diag(g.C.U)) + 0.5n * log(2π)
end

"""
    fit(g::ExactGP; restarts=1, ad=AutoForwardDiff()) -> ExactGP

Optimize the kernel lengthscale `ℓ` and signal variance `σ_f²` by minimizing
[`nlml`](@ref) with LBFGS, returning a GP re-conditioned at the best hyperparameters.

Optimization runs in log-space (`[logℓ, logσ²]`, bounded to `[-6, 6]`) so both stay
positive. The recovered kernel is `σ_f²·with_lengthscale(SqExponentialKernel(), ℓ)`.
Fitting `σ_f²` (not just `ℓ`) calibrates the function scale, which the derivative/straddle
acquisitions need — a unit-variance prior miscalibrates them on any non-unit-scale target.
With `restarts > 1`, extra runs start from the initial point jittered in log-space and the
lowest-NLML result wins. `ad` selects the DifferentiationInterface backend for the gradient.

!!! note
    v1 assumes the prior kernel is a (scaled) `with_lengthscale(SqExponentialKernel(), ℓ)`.
"""
function fit(g::ExactGP; restarts::Int=1, ad=AutoForwardDiff())
    @assert _basekernel(g.prior.kernel) isa SqExponentialKernel "v1 fit assumes a (scaled) with_lengthscale(SqExponentialKernel(), ℓ)"
    X = g.x; y = g.δ .+ AbstractGPs.mean(g.prior, g.x)
    noise = g.noise; meanfn = g.prior.mean
    p0 = [log(_lengthscale(g.prior.kernel)), log(_outputscale(g.prior.kernel))]
    # p = [logℓ, logσ²]; closure stays AD-compatible (no ParameterHandling unflatten).
    mkkernel(p) = exp(p[2]) * with_lengthscale(SqExponentialKernel(), exp(p[1]))
    loss(p, _) = nlml(update(ExactGP(mkkernel(p); noise=noise, mean=meanfn), X, y))
    best = g; best_nlml = nlml(g)
    for r in 1:restarts
        start = r == 1 ? p0 : p0 .+ 0.1 .* randn(2)              # first run exact, rest jittered
        prob = OptimizationProblem(OptimizationFunction(loss, ad), start; lb=[-6.0, -6.0], ub=[6.0, 6.0])
        sol = solve(prob, LBFGS())
        gp_cand = update(ExactGP(mkkernel(sol.u); noise=noise, mean=meanfn), X, y)
        if nlml(gp_cand) < best_nlml; best, best_nlml = gp_cand, nlml(gp_cand); end
    end
    return best
end
