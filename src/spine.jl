"""
    AbstractGPModel <: AbstractGPs.AbstractGP

Contract for a GP that plugs into Magpie's active-learning loop and SciML bridge.

Subtyping `AbstractGPs.AbstractGP` inherits the Distributions interface (`f(x)` →
`FiniteGP`, `rand`, `logpdf`, `mean`/`var`/`cov`). On top of that, an
`AbstractGPModel` is expected to implement the extended contract the loop drives:

  - `update(g, X, y)` — incremental conditioning on new observations
  - `fit(g; …)` — hyperparameter optimization
  - `predmean(g, u)` — posterior mean at a single point

Subtype this and implement those methods to plug in a new GP (e.g. a sparse GP).
Bundled implementations: [`ExactGP`](@ref), [`LaplaceGP`](@ref).
"""
abstract type AbstractGPModel <: AbstractGPs.AbstractGP end

@doc raw"""
    ExactGP{Tp,Tx,Tδ,TC,Tα} <: AbstractGPModel

Exact-regression GP holding the cached state of an incremental conditioning. The
posterior mean reuses the cached weights ``\alpha``:

```math
\mu(x_\ast) = m(x_\ast) + \mathrm{cov}(x_\ast, x)\,\alpha,
\qquad \alpha = C^{-1}\delta.
```

Field names follow Rasmussen & Williams (Algorithm 2.1).

# Fields
  - `prior`: the prior `AbstractGPs.GP` (mean + kernel)
  - `x`: training inputs
  - `δ`: residuals ``\delta = y - m(x)`` against the prior mean
  - `C`: Cholesky factorization of ``K + \sigma^2 I``, extended in place on each `update`
  - `α`: cached weights ``\alpha = C^{-1}\delta``
  - `noise`: observation-noise variance ``\sigma^2`` added to the kernel diagonal

# Constructor

    ExactGP(kernel::Kernel; noise=1e-6, mean=AbstractGPs.ZeroMean())

Build an unconditioned `ExactGP`; condition it on data with [`update`](@ref).
"""
struct ExactGP{Tp,Tx,Tδ,TC,Tα} <: AbstractGPModel
    prior::Tp; x::Tx; δ::Tδ; C::TC; α::Tα; noise::Float64
end
ExactGP(kernel::Kernel; noise::Real=1e-6, mean=AbstractGPs.ZeroMean()) =
    ExactGP(AbstractGPs.GP(mean, kernel), Any[], Float64[], nothing, Float64[], Float64(noise))

# True once the GP has been conditioned on data (the Cholesky factor exists).
_hasdata(g::ExactGP) = g.C !== nothing

"""
    _chol(K)

Cholesky factorization of a symmetric matrix — the single factorization chokepoint.

`Symmetric(K)` is AD-neutral under Mooncake (`cholesky` routes to `LAPACK.potrf!`
regardless of the wrapper; #414 is a ChainRules-only bug AbstractGPs already lives
with). `check=false` tolerates roundoff-induced tiny-negative pivots when [`fit`](@ref)
probes extreme lengthscales under dual numbers. For an Enzyme/ChainRules backend,
re-add `Matrix(...)` via a per-backend method here.
"""
_chol(K) = cholesky(Symmetric(K); check=false)

"""Posterior mean at `xs`: prior mean `m(xs)` plus `cov(xs, x)·α` once conditioned."""
function Statistics.mean(g::ExactGP, xs::AbstractVector)
    m = AbstractGPs.mean(g.prior, xs)
    _hasdata(g) ? m .+ AbstractGPs.cov(g.prior, xs, g.x) * g.α : m
end

"""Posterior (marginal) variance at `xs`: prior variance minus the data-explained part."""
function Statistics.var(g::ExactGP, xs::AbstractVector)
    v = AbstractGPs.var(g.prior, xs)
    _hasdata(g) ? v .- diag_Xt_invA_X(g.C, AbstractGPs.cov(g.prior, g.x, xs)) : v
end

# Posterior cross-covariance between two input sets `xs` and `ys`.
function Statistics.cov(g::ExactGP, xs::AbstractVector, ys::AbstractVector)
    c = AbstractGPs.cov(g.prior, xs, ys)
    _hasdata(g) ? c .- Xt_invA_Y(AbstractGPs.cov(g.prior, g.x, xs), g.C, AbstractGPs.cov(g.prior, g.x, ys)) : c
end

# Posterior covariance matrix within a single input set `xs`.
function Statistics.cov(g::ExactGP, xs::AbstractVector)
    c = AbstractGPs.cov(g.prior, xs)
    _hasdata(g) ? c .- Xt_invA_X(g.C, AbstractGPs.cov(g.prior, g.x, xs)) : c
end

"""
    update(g::ExactGP, X, y) -> ExactGP

Condition the GP on observations `(X, y)`, returning a new conditioned `ExactGP`.

The first call factorizes `K + σ²I` from scratch; later calls extend the existing
Cholesky factor incrementally via `AbstractGPs.update_chol`. A scalar `y` conditions
on a single point.
"""
function update(g::ExactGP, X::AbstractVector, y::AbstractVector)
    _hasdata(g) && return _update_incremental(g, X, y)
    xnew = collect(X)
    δnew = y .- AbstractGPs.mean(g.prior, xnew)
    K = AbstractGPs.cov(g.prior, xnew) + g.noise * I
    C = _chol(K)
    ExactGP(g.prior, xnew, δnew, C, C \ δnew, g.noise)
end
update(g::ExactGP, x, y::Real) = update(g, [x], [y])

"""
    mean_and_var(g::ExactGP, xs) -> (mean, var)

Posterior mean and marginal variance at `xs` in one pass, sharing the
cross-covariance `cov(x, xs)` between both — cheaper than separate calls.
"""
function mean_and_var(g::ExactGP, xs::AbstractVector)
    m = AbstractGPs.mean(g.prior, xs)
    _hasdata(g) || return (m, AbstractGPs.var(g.prior, xs))
    Ks = AbstractGPs.cov(g.prior, g.x, xs)                    # shared between mean and variance
    return (m .+ Ks' * g.α, AbstractGPs.var(g.prior, xs) .- diag_Xt_invA_X(g.C, Ks))
end

"""
    predict(g, xs) -> (mean, var)

Posterior mean and marginal variance at the inputs `xs` (an alias for
`mean_and_var` on any `AbstractGP`).
"""
predict(g::AbstractGPs.AbstractGP, xs::AbstractVector) = mean_and_var(g, xs)

"""
    predmean(g::ExactGP, u) -> Real

Posterior mean at a single input `u`, returned as a scalar (unlike `mean(g, [u])`,
which returns a length-1 vector).
"""
predmean(g::ExactGP, u) = _hasdata(g) ?
    only(AbstractGPs.mean(g.prior, [u])) + dot(AbstractGPs.cov(g.prior, g.x, [u]), g.α) :
    only(AbstractGPs.mean(g.prior, [u]))

"""
    _update_incremental(g::ExactGP, X, y) -> ExactGP

Extend an already-conditioned GP with new data using a block Cholesky update,
growing `C` from `n×n` to `(n+m)×(n+m)` via `AbstractGPs.update_chol` rather than
refactorizing. `C12` is the cross-covariance to the existing points and `C22` the
covariance among the new points.
"""
function _update_incremental(g::ExactGP, X::AbstractVector, y::AbstractVector)
    xnew = collect(X)
    C12 = AbstractGPs.cov(g.prior, g.x, xnew)                 # cross-cov, old × new
    C22 = Matrix(Symmetric(AbstractGPs.cov(g.prior, xnew) + g.noise * I))
    Cext = update_chol(g.C, C12, C22)
    xall = vcat(g.x, xnew)
    δall = vcat(g.δ, y .- AbstractGPs.mean(g.prior, xnew))
    ExactGP(g.prior, xall, δall, Cext, Cext \ δall, g.noise)
end
