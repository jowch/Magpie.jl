# Our GP types subtype AbstractGPs.AbstractGP (so we inherit f(x)→FiniteGP, rand,
# logpdf, the Distributions interface) but additionally implement an extended contract
# the active-learning loop and SciML bridge require: `update(g, X, y)` (incremental
# conditioning), `fit(g; …)`, `predmean(g, u)`. `AbstractGPModel` names exactly that contract —
# subtype it + implement those methods to plug a new GP (e.g. a sparse GP) into the loop.
abstract type AbstractGPModel <: AbstractGPs.AbstractGP end

struct ExactGP{Tp,Tx,Tδ,TC,Tα} <: AbstractGPModel
    prior::Tp; x::Tx; δ::Tδ; C::TC; α::Tα; noise::Float64
end
ExactGP(kernel::Kernel; noise::Real=1e-6, mean=AbstractGPs.ZeroMean()) =
    ExactGP(AbstractGPs.GP(mean, kernel), Any[], Float64[], nothing, Float64[], Float64(noise))
_hasdata(g::ExactGP) = g.C !== nothing

# Central factorization chokepoint. `Symmetric(K)` is AD-neutral under Mooncake
# (cholesky routes to LAPACK.potrf! regardless of wrapper; #414 is a ChainRules-only
# bug AbstractGPs already lives with). `check=false` tolerates roundoff-induced
# tiny-negative pivots when `fit` probes extreme lengthscales under Duals.
# For an Enzyme/ChainRules backend, re-add Matrix(...) via a per-backend method here.
_chol(K) = cholesky(Symmetric(K); check=false)

function Statistics.mean(g::ExactGP, xs::AbstractVector)
    m = AbstractGPs.mean(g.prior, xs)
    _hasdata(g) ? m .+ AbstractGPs.cov(g.prior, xs, g.x) * g.α : m
end
function Statistics.var(g::ExactGP, xs::AbstractVector)
    v = AbstractGPs.var(g.prior, xs)
    _hasdata(g) ? v .- diag_Xt_invA_X(g.C, AbstractGPs.cov(g.prior, g.x, xs)) : v
end
function Statistics.cov(g::ExactGP, xs::AbstractVector, ys::AbstractVector)
    c = AbstractGPs.cov(g.prior, xs, ys)
    _hasdata(g) ? c .- Xt_invA_Y(AbstractGPs.cov(g.prior, g.x, xs), g.C, AbstractGPs.cov(g.prior, g.x, ys)) : c
end
function Statistics.cov(g::ExactGP, xs::AbstractVector)
    c = AbstractGPs.cov(g.prior, xs)
    _hasdata(g) ? c .- Xt_invA_X(g.C, AbstractGPs.cov(g.prior, g.x, xs)) : c
end

function update(g::ExactGP, X::AbstractVector, y::AbstractVector)
    _hasdata(g) && return _update_incremental(g, X, y)        # Task 2
    xnew = collect(X)
    δnew = y .- AbstractGPs.mean(g.prior, xnew)
    K = AbstractGPs.cov(g.prior, xnew) + g.noise * I
    C = _chol(K)
    ExactGP(g.prior, xnew, δnew, C, C \ δnew, g.noise)
end
update(g::ExactGP, x, y::Real) = update(g, [x], [y])

function mean_and_var(g::ExactGP, xs::AbstractVector)
    m = AbstractGPs.mean(g.prior, xs)
    _hasdata(g) || return (m, AbstractGPs.var(g.prior, xs))
    Ks = AbstractGPs.cov(g.prior, g.x, xs)                    # computed once
    return (m .+ Ks' * g.α, AbstractGPs.var(g.prior, xs) .- diag_Xt_invA_X(g.C, Ks))
end
predict(g::AbstractGPs.AbstractGP, xs::AbstractVector) = mean_and_var(g, xs)
predmean(g::ExactGP, u) = _hasdata(g) ?
    only(AbstractGPs.mean(g.prior, [u])) + dot(AbstractGPs.cov(g.prior, g.x, [u]), g.α) :
    only(AbstractGPs.mean(g.prior, [u]))

function _update_incremental(g::ExactGP, X::AbstractVector, y::AbstractVector)
    xnew = collect(X)
    C12 = AbstractGPs.cov(g.prior, g.x, xnew)                 # (n × m)
    C22 = Matrix(Symmetric(AbstractGPs.cov(g.prior, xnew) + g.noise * I))
    Cext = update_chol(g.C, C12, C22)
    xall = vcat(g.x, xnew)
    δall = vcat(g.δ, y .- AbstractGPs.mean(g.prior, xnew))
    ExactGP(g.prior, xall, δall, Cext, Cext \ δall, g.noise)
end
