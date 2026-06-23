using Optimization, OptimizationOptimJL, DifferentiationInterface
using Optimisers: destructure

# Peel ScaledKernel/TransformedKernel wrappers to read hyperparameters.
# ℓ from with_lengthscale(k, ℓ) == k ∘ ScaleTransform(1/ℓ); σ² from `c * k` == ScaledKernel.
_lengthscale(k::KernelFunctions.ScaledKernel) = _lengthscale(k.kernel)
_lengthscale(k::KernelFunctions.TransformedKernel) = _ls_from_transform(k.transform)
_lengthscale(k) = throw(
    ArgumentError(
        "no scalar lengthscale for a kernel of type $(nameof(typeof(k))); grad_predict's analytic " *
            "prior-gradient-variance and LocalPenalization's radius require an isotropic " *
            "`with_lengthscale` kernel (optionally scaled).",
    ),
)
_ls_from_transform(t::KernelFunctions.ScaleTransform) = 1 / only(t.s)
_ls_from_transform(t) = throw(
    ArgumentError(
        "no scalar lengthscale for an ARD/$(nameof(typeof(t))) transform; grad_predict and " *
            "LocalPenalization are scalar-lengthscale only.",
    ),
)
_outputscale(k) = 1.0
_outputscale(k::KernelFunctions.ScaledKernel) = only(k.σ²) * _outputscale(k.kernel)
_basekernel(k) = k
_basekernel(k::KernelFunctions.TransformedKernel) = _basekernel(k.kernel)
_basekernel(k::KernelFunctions.ScaledKernel) = _basekernel(k.kernel)


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

Optimize the kernel's hyperparameters by minimizing [`nlml`](@ref) (plus a lengthscale prior;
see below) with LBFGS, returning a GP re-conditioned at the best hyperparameters.

`fit` treats the GP's kernel as a structural template: it `Optimisers.destructure`s it into a
flat vector of positive scale parameters (inverse-lengthscales and output scales), optimizes the
**log** of that vector (bounded to `[-6, 6]`, so every scale stays positive), and rebuilds the
kernel. This supports **any** KernelFunctions kernel whose hyperparameters are positive scales:
`SqExponential`/`Matern`/`RationalQuadratic` bases, **ARD** (a lengthscale per input dimension),
and **sums/products** of these. A bare `with_lengthscale(base, ℓ)` kernel (no signal-variance
factor) is auto-wrapped as `1.0 * k` so σ_f² is always a tunable leaf — fitting σ_f² calibrates
the function scale, which the derivative/straddle acquisitions need.

`ad` selects the DifferentiationInterface backend; `nothing` (default) picks automatically:
`AutoForwardDiff()` for `SqExponentialKernel` bases (fast, smooth at `r=0`) and `AutoMooncake()`
otherwise (Matérn bases NaN under ForwardDiff at coincident points via `sqrt(0)`; composites
default to Mooncake too). With `restarts > 1`, extra runs start from the initial point jittered
in log-space and the result with the lowest **penalized** objective wins.

## Lengthscale prior (MAP, default on — scalar-lengthscale kernels only)

For a single-scalar-lengthscale kernel, `fit` is MAP: it adds a weakly-informative Gaussian prior
on `logℓ`, `0.5·((logℓ − μ)/σ)²`. **`ℓ_prior=:auto`** (default) centres it on the **initial
lengthscale** of `g`'s kernel with width `σ=0.75` (log units) — *refine the lengthscale you
specified, don't run away from it.* This matters when data is scarce: pure MLE drives `ℓ` up (a
flat surface explains few points cheaply), over-smoothing away the wells/saddles one hunts. Pass
`ℓ_prior=(μ, σ)` to set centre/width explicitly, or `ℓ_prior=nothing` for pure MLE. σ_f² is never
penalized.

For **ARD or composite** kernels there is no single lengthscale, so `ℓ_prior=:auto` falls back to
**no prior** (pure MLE); passing an explicit `ℓ_prior=(μ,σ)` then raises an `ArgumentError`.

!!! note
    `fit` requires the kernel's hyperparameters to be **positive scales** (lengthscales, output
    scales). Kernels with non-positive or non-scale leaves (e.g. `LinearKernel`, whose offset
    destructures to `0.0`) raise an `ArgumentError`.
"""
function fit(g::ExactGP; restarts::Int = 1, ad = nothing, ℓ_prior = :auto)
    g.d == 1 ||
        throw(ArgumentError("fit currently supports single-output GPs (d=1); got d=$(g.d)."))
    # Auto-wrap so a tunable σ_f² leaf is always present (a bare `with_lengthscale` has none).
    # Don't wrap composite (sum/product) kernels — their components already carry σ_f² leaves.
    k0 =
        g.prior.kernel isa KernelFunctions.ScaledKernel ||
        g.prior.kernel isa KernelFunctions.KernelSum ||
        g.prior.kernel isa KernelFunctions.KernelProduct ? g.prior.kernel : 1.0 * g.prior.kernel
    θ0, re = destructure(k0)
    (!isempty(θ0) && all(>(0), θ0)) || throw(
        ArgumentError(
            "fit optimizes positive scale hyperparameters (lengthscales, output scales) in log-space, " *
                "but the kernel destructured to $(θ0) — empty or with a non-positive leaf. Supported: " *
                "SqExponential/Matern/RationalQuadratic kernels (incl. ARD) and their sums/products. " *
                "Got $(typeof(g.prior.kernel)).",
        )
    )
    ad === nothing && (ad = _default_ad(k0))
    X = g.x; y = g.δ .+ AbstractGPs.mean(g.prior, g.x)
    noise = g.noise; meanfn = g.prior.mean
    # MAP lengthscale prior: scalar-lengthscale kernels only (`_lengthscale` throws otherwise).
    scalar_ℓ = try
        (_lengthscale(k0); true)
    catch
        false
    end
    if ℓ_prior === :auto
        pri = scalar_ℓ ? (log(_lengthscale(k0)), 0.75) : nothing
    elseif ℓ_prior === nothing
        pri = nothing
    else
        scalar_ℓ || throw(
            ArgumentError(
                "ℓ_prior=(μ,σ) requires a scalar-lengthscale kernel; this one is ARD/composite. Use ℓ_prior=nothing.",
            )
        )
        pri = ℓ_prior
    end
    logθ0 = log.(θ0)
    np = length(logθ0)
    # logθ → kernel via the rebuild closure; penalty reads logℓ off the rebuilt kernel (AD-safe).
    function loss(logθ, _)
        k = re(exp.(logθ))
        base = nlml(update(ExactGP(k; noise = noise, mean = meanfn), X, y))
        pen = pri === nothing ? zero(eltype(logθ)) : 0.5 * ((log(_lengthscale(k)) - pri[1]) / pri[2])^2
        return base + pen
    end
    obj(logθ) = loss(logθ, nothing)
    best = g; best_obj = obj(logθ0)                                  # penalty(logθ0)=0 for :auto
    for r in 1:restarts
        start = r == 1 ? logθ0 : logθ0 .+ 0.1 .* randn(np)           # first run exact, rest jittered
        prob = OptimizationProblem(OptimizationFunction(loss, ad), start; lb = fill(-6.0, np), ub = fill(6.0, np))
        sol = solve(prob, LBFGS())
        if obj(sol.u) < best_obj
            best = update(ExactGP(re(exp.(sol.u)); noise = noise, mean = meanfn), X, y)
            best_obj = obj(sol.u)
        end
    end
    return best
end
