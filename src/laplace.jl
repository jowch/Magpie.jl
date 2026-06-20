struct LaplaceGP{Tp,Tx,Ty,Ta,TW,TL} <: AbstractGPModel
    prior::Tp; x::Tx; y::Ty; a::Ta; W::TW; L::TL   # y::Vector{Bool} stored for incremental conditioning; a = K⁻¹(f̂−m); W, L = chol(B) from MAP
end
LaplaceGP(kernel::Kernel; mean=AbstractGPs.ZeroMean()) =
    LaplaceGP(AbstractGPs.GP(mean, kernel), Any[], Bool[], Float64[], Float64[], nothing)
_hasdata(g::LaplaceGP) = g.L !== nothing
_σ(z) = 1 / (1 + exp(-z))

function _laplace_fit(prior, x, y_bool)
    m = AbstractGPs.mean(prior, x); t = float.(y_bool)
    K = Matrix(Symmetric(AbstractGPs.cov(prior, x))) + 1e-9I
    f = copy(m); local a, W, L
    for _ in 1:30                                   # unrolled, fixed count (Mooncake-clean)
        π_ = _σ.(f); W = π_ .* (1 .- π_); sW = sqrt.(W)
        L = _chol(I + (sW * sW') .* K).L
        b = W .* (f .- m) .+ (t .- π_)
        a = b .- sW .* (L' \ (L \ (sW .* (K * b))))
        f = K * a .+ m
    end
    return a, W, L
end

function update(g::LaplaceGP, X::AbstractVector, y::AbstractVector{Bool})
    # Accumulate all data (LaplaceGP re-fits from scratch; retain history for incremental conditioning)
    xall = vcat(g.x, collect(X))
    yall = vcat(g.y, y)
    a, W, L = _laplace_fit(g.prior, xall, yall)
    LaplaceGP(g.prior, xall, yall, a, W, L)
end

function _latent_moments(g::LaplaceGP, xs)
    Ks = AbstractGPs.cov(g.prior, g.x, xs); sW = sqrt.(g.W)
    μ = AbstractGPs.mean(g.prior, xs) .+ Ks' * g.a
    v = g.L \ (sW .* Ks)
    σ² = AbstractGPs.var(g.prior, xs) .- vec(sum(v .^ 2; dims=1))
    return μ, σ²
end
Statistics.mean(g::LaplaceGP, xs::AbstractVector) = _hasdata(g) ? _latent_moments(g, xs)[1] : AbstractGPs.mean(g.prior, xs)
Statistics.var(g::LaplaceGP, xs::AbstractVector)  = _hasdata(g) ? _latent_moments(g, xs)[2] : AbstractGPs.var(g.prior, xs)
mean_and_var(g::LaplaceGP, xs::AbstractVector) = _hasdata(g) ? _latent_moments(g, xs) : (AbstractGPs.mean(g.prior,xs), AbstractGPs.var(g.prior,xs))
function Statistics.cov(g::LaplaceGP, xs::AbstractVector, ys::AbstractVector)
    c = AbstractGPs.cov(g.prior, xs, ys)
    _hasdata(g) || return c
    sW = sqrt.(g.W)
    vx = g.L \ (sW .* AbstractGPs.cov(g.prior, g.x, xs)); vy = g.L \ (sW .* AbstractGPs.cov(g.prior, g.x, ys))
    return c .- vx' * vy
end
Statistics.cov(g::LaplaceGP, xs::AbstractVector) =
    Diagonal(_hasdata(g) ? _latent_moments(g, xs)[2] : AbstractGPs.var(g.prior, xs))
predmean(g::LaplaceGP, u) = mean(g, [u])[1]
fit(g::LaplaceGP; kwargs...) = g  # v1: no hyperparameter refit for the Laplace path
