struct LaplaceGP{Tp,Tx,Ta,TW,TL} <: AbstractGPs.AbstractGP
    prior::Tp; x::Tx; a::Ta; W::TW; L::TL          # a = K⁻¹(f̂−m); W, L = chol(B) from MAP
end
LaplaceGP(kernel::Kernel; mean=AbstractGPs.ZeroMean()) =
    LaplaceGP(AbstractGPs.GP(mean, kernel), Any[], Float64[], Float64[], nothing)
_hasdata(g::LaplaceGP) = g.L !== nothing
_σ(z) = 1 / (1 + exp(-z))

function update(g::LaplaceGP, X::AbstractVector, y::AbstractVector{Bool})
    x = collect(X); m = AbstractGPs.mean(g.prior, x); t = float.(y)
    K = Matrix(Symmetric(AbstractGPs.cov(g.prior, x))) + 1e-9I
    f = copy(m); local a, W, L
    for _ in 1:30                                   # unrolled, fixed count (Mooncake-clean)
        π_ = _σ.(f); W = π_ .* (1 .- π_); sW = sqrt.(W)
        L = _chol(I + (sW * sW') .* K).L
        b = W .* (f .- m) .+ (t .- π_)
        a = b .- sW .* (L' \ (L \ (sW .* (K * b))))
        f = K * a .+ m
    end
    LaplaceGP(g.prior, x, a, W, L)
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
Statistics.cov(g::LaplaceGP, xs::AbstractVector) = (μσ = _latent_moments(g, xs); Diagonal(_hasdata(g) ? μσ[2] : AbstractGPs.var(g.prior, xs)))
predmean(g::LaplaceGP, u) = mean(g, [u])[1]
fit(g::LaplaceGP; kwargs...) = g  # v1: no hyperparameter refit for the Laplace path
