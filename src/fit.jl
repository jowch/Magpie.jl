using Optimization, OptimizationOptimJL, DifferentiationInterface

# lengthscale of with_lengthscale(SqExponentialKernel(), ℓ) == SqExp ∘ ScaleTransform(1/ℓ)
_lengthscale(k) = 1 / only(k.transform.s)

function nlml(g::ExactGP)
    _hasdata(g) || return 0.0
    n = length(g.δ)
    return 0.5*dot(g.δ, g.α) + sum(log, diag(g.C.U)) + 0.5n*log(2π)
end

function fit(g::ExactGP; restarts::Int=1, ad=AutoForwardDiff())
    @assert g.prior.kernel isa KernelFunctions.TransformedKernel "v1 fit assumes with_lengthscale(SqExponentialKernel(), ℓ)"
    X = g.x; y = g.δ .+ AbstractGPs.mean(g.prior, g.x)
    noise = g.noise
    # Store initial log-lengthscale; optimize in log-space (unconstrained), recover ℓ = exp(logℓ)
    logℓ0 = log(_lengthscale(g.prior.kernel))
    # loss takes a length-1 vector [logℓ] — AD-compatible (no ParameterHandling unflatten inside)
    loss(flat, _) = nlml(update(ExactGP(with_lengthscale(SqExponentialKernel(), exp(only(flat))); noise=noise), X, y))
    best = g; bestv = nlml(g)
    for r in 1:restarts
        f0 = r == 1 ? [logℓ0] : [logℓ0 + 0.1*randn()]
        prob = OptimizationProblem(OptimizationFunction(loss, ad), f0; lb=[-6.0], ub=[6.0])
        sol = solve(prob, LBFGS())
        cand = update(ExactGP(with_lengthscale(SqExponentialKernel(), exp(only(sol.u))); noise=noise), X, y)
        if nlml(cand) < bestv; best, bestv = cand, nlml(cand); end
    end
    return best
end
