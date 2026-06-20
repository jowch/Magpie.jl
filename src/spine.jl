struct ExactGP{Tp,Tx,Tδ,TC,Tα} <: AbstractGPs.AbstractGP
    prior::Tp; x::Tx; δ::Tδ; C::TC; α::Tα; noise::Float64
end
ExactGP(kernel::Kernel; noise::Real=1e-6, mean=AbstractGPs.ZeroMean()) =
    ExactGP(AbstractGPs.GP(mean, kernel), Any[], Float64[], nothing, Float64[], Float64(noise))
_hasdata(g::ExactGP) = g.C !== nothing

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
    C = cholesky(Matrix(Symmetric(K)))
    ExactGP(g.prior, xnew, δnew, C, C \ δnew, g.noise)
end
update(g::ExactGP, x, y::Real) = update(g, [x], [y])

_jitter(C22::AbstractMatrix; rel=1e-10) = rel * (tr(C22) / size(C22, 1))
function _update_incremental(g::ExactGP, X::AbstractVector, y::AbstractVector)
    xnew = collect(X)
    C12 = AbstractGPs.cov(g.prior, g.x, xnew)                 # (n × m)
    C22 = Matrix(Symmetric(AbstractGPs.cov(g.prior, xnew) + g.noise * I))
    Cext = update_chol(g.C, C12, C22)
    xall = vcat(g.x, xnew)
    δall = vcat(g.δ, y .- AbstractGPs.mean(g.prior, xnew))
    ExactGP(g.prior, xall, δall, Cext, Cext \ δall, g.noise)
end
