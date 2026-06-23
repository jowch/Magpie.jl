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

_kernelfamily(::SqExponentialKernel) = SqExponentialKernel()
_kernelfamily(::Matern32Kernel) = Matern32Kernel()
_kernelfamily(::Matern52Kernel) = Matern52Kernel()
_kernelfamily(k) = throw(ArgumentError("fit supports SqExponential/Matern32/Matern52 base kernels; got $(typeof(k)). (grad_predict's derivative path covers the same set.)"))

# AutoForwardDiff is fast and correct for SqExponential; Matérn kernels NaN under ForwardDiff at
# coincident points (sqrt(0) non-differentiable), so default them to Mooncake (the project's
# Mooncake-first backend), which differentiates the r=0 diagonal correctly.
_default_ad(k) = _basekernel(k) isa SqExponentialKernel ? AutoForwardDiff() : AutoMooncake(; config = nothing)

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
    n = size(g.δ, 1)                                   # rows = #points (length for a Vector)
    return 0.5 * dot(g.δ, g.α) + g.d * (sum(log, diag(g.C.U)) + 0.5n * log(2π))
end

"""
    fit(g::ExactGP; restarts=1, ad=nothing, ℓ_prior=:auto) -> ExactGP

Optimize the kernel lengthscale `ℓ` and signal variance `σ_f²` by minimizing
[`nlml`](@ref) (plus a lengthscale prior; see below) with LBFGS, returning a GP
re-conditioned at the best hyperparameters.

Optimization runs in log-space (`[logℓ, logσ²]`, bounded to `[-6, 6]`) so both stay
positive. The recovered kernel is `σ_f²·with_lengthscale(SqExponentialKernel(), ℓ)`.
Fitting `σ_f²` (not just `ℓ`) calibrates the function scale, which the derivative/straddle
acquisitions need — a unit-variance prior miscalibrates them on any non-unit-scale target.
With `restarts > 1`, extra runs start from the initial point jittered in log-space and the
result with the lowest **penalized** objective wins. `ad` selects the DifferentiationInterface
backend for the gradient; `nothing` (the default) picks automatically: `AutoForwardDiff()` for
`SqExponentialKernel` (fast, no `sqrt(0)` issue) and `AutoMooncake()` for Matérn kernels
(ForwardDiff produces NaN at coincident training points via `sqrt(0)`).

## Lengthscale prior (MAP, default on)

`fit` is MAP, not pure MLE: it adds a weakly-informative Gaussian prior on `logℓ` to the
objective, `0.5·((logℓ − μ)/σ)²`. **`ℓ_prior=:auto`** (the default) centres that prior on the
**initial lengthscale** of `g`'s kernel with width `σ=0.75` (log units), i.e. *refine the
lengthscale you specified, don't run away from it.* This matters when data is scarce: pure MLE
drives `ℓ` **up** (a flat surface explains few points cheaply), over-smoothing away the very
features (wells, saddles) one is hunting — verified to collapse critical-point recovery at small
`n`. The initial `ℓ` you chose encodes the feature scale you expect, so anchoring to it (softly)
keeps `fit` well-behaved. With enough data the likelihood dominates the prior and recovers the
data-driven `ℓ` as usual.

Pass `ℓ_prior=(μ, σ)` to set the prior centre/width in log-space explicitly, or
`ℓ_prior=nothing` for pure MLE (the pre-MAP behaviour). `σ_f²` is never penalized.

!!! note
    `fit` supports `SqExponentialKernel`, `Matern32Kernel`, and `Matern52Kernel` base kernels
    (the same set `grad_predict`'s derivative path covers). Throws `ArgumentError` for other families.
"""
function fit(g::ExactGP; restarts::Int = 1, ad = nothing, ℓ_prior = :auto)
    g.d == 1 ||
        throw(ArgumentError("fit currently supports single-output GPs (d=1); got d=$(g.d)."))
    ad === nothing && (ad = _default_ad(g.prior.kernel))
    fam = _kernelfamily(_basekernel(g.prior.kernel))     # validates + returns a fresh base kernel of the same family
    X = g.x; y = g.δ .+ AbstractGPs.mean(g.prior, g.x)
    noise = g.noise; meanfn = g.prior.mean
    logℓ0 = log(_lengthscale(g.prior.kernel))
    p0 = [logℓ0, log(_outputscale(g.prior.kernel))]
    pri = ℓ_prior === :auto ? (logℓ0, 0.75) : ℓ_prior            # (μ, σ) on logℓ, or nothing
    penalty(p) = pri === nothing ? zero(eltype(p)) : 0.5 * ((p[1] - pri[1]) / pri[2])^2
    # p = [logℓ, logσ²]; closure stays AD-compatible (no ParameterHandling unflatten).
    mkkernel(p) = exp(p[2]) * with_lengthscale(fam, exp(p[1]))
    loss(p, _) = nlml(update(ExactGP(mkkernel(p); noise = noise, mean = meanfn), X, y)) + penalty(p)
    obj(p) = loss(p, nothing)
    best = g; best_obj = obj(p0)                                      # penalty(p0)=0 for :auto
    for r in 1:restarts
        start = r == 1 ? p0 : p0 .+ 0.1 .* randn(2)              # first run exact, rest jittered
        prob = OptimizationProblem(OptimizationFunction(loss, ad), start; lb = [-6.0, -6.0], ub = [6.0, 6.0])
        sol = solve(prob, LBFGS())
        obj(sol.u) < best_obj && ((best, best_obj) = (update(ExactGP(mkkernel(sol.u); noise = noise, mean = meanfn), X, y), obj(sol.u)))
    end
    return best
end
