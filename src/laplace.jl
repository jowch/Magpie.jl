@doc raw"""
    LaplaceGP{Tp,Tx,Ty,Ta,TW,TL} <: AbstractGPModel

Binary-classification GP using the Laplace approximation (Rasmussen & Williams,
Algorithm 3.1) with a logistic likelihood. The latent posterior is approximated by a
Gaussian centred at the MAP ``\hat{f}`` with precision

```math
\Sigma^{-1} = K^{-1} + W, \qquad B = I + \sqrt{W}\,K\,\sqrt{W},
```

where ``W`` is the (diagonal) likelihood Hessian and ``B`` is the well-conditioned
matrix actually factorized. The MAP is re-fit from scratch on each `update`, so the
full observation history is retained. Field names follow R&W Algorithm 3.1.

# Fields
  - `prior`: the prior `AbstractGPs.GP` over the latent function `f`
  - `x`: training inputs
  - `y`: `Bool` class labels, kept so re-fitting can use the full history
  - `a`: dual vector ``K^{-1}(\hat{f} - m)`` at the MAP latent ``\hat{f}``
  - `W`: diagonal of the likelihood Hessian, ``W_i = \hat{\pi}_i(1 - \hat{\pi}_i)``, at the MAP
  - `L`: Cholesky factor of ``B = I + \sqrt{W}\,K\,\sqrt{W}`` at the MAP

# Constructor

    LaplaceGP(kernel::Kernel; mean=AbstractGPs.ZeroMean())

Build an unconditioned `LaplaceGP`; condition it on labels with [`update`](@ref).
"""
struct LaplaceGP{Tp, Tx, Ty, Ta, TW, TL} <: AbstractGPModel
    prior::Tp; x::Tx; y::Ty; a::Ta; W::TW; L::TL
end
LaplaceGP(kernel::Kernel; mean = AbstractGPs.ZeroMean()) =
    LaplaceGP(AbstractGPs.GP(mean, kernel), Any[], Bool[], Float64[], Float64[], nothing)

# True once the GP has been conditioned on data (the MAP Cholesky factor exists).
_hasdata(g::LaplaceGP) = g.L !== nothing

# Logistic link σ(z) = 1/(1 + e⁻ᶻ), mapping a latent value to a class probability.
_σ(z) = 1 / (1 + exp(-z))

"""
    _laplace_fit(prior, x, y_bool) -> (a, W, L)

Find the MAP latent `f̂` for a logistic likelihood by Newton iteration (Rasmussen &
Williams, Algorithm 3.1), and return the posterior quantities cached by [`LaplaceGP`](@ref):
the dual `a`, the Hessian diagonal `W`, and the Cholesky factor `L` of `I + √W·K·√W`.

The loop count is fixed and unrolled (no convergence branch) to stay Mooncake-clean
for autodiff. `t` is the float view of the labels, `m` the prior mean, and `π_` the
current class probabilities `σ(f)`.
"""
function _laplace_fit(prior, x, y_bool)
    m = AbstractGPs.mean(prior, x); t = float.(y_bool)
    K = Matrix(Symmetric(AbstractGPs.cov(prior, x))) + 1.0e-9I
    f = copy(m); local a, W, L
    for _ in 1:30                                   # fixed, unrolled Newton steps (Mooncake-clean)
        π_ = _σ.(f); W = π_ .* (1 .- π_); sW = sqrt.(W)
        L = _chol(I + (sW * sW') .* K).L
        b = W .* (f .- m) .+ (t .- π_)
        a = b .- sW .* (L' \ (L \ (sW .* (K * b))))
        f = K * a .+ m
    end
    return a, W, L
end

function _to_labels(y)
    eltype(y) === Bool && return collect(Bool, y)
    vals = unique(y)
    Set(vals) ⊆ Set((0, 1)) && return Bool[yi == 1 for yi in y]
    Set(vals) ⊆ Set((-1, 1)) && return Bool[yi == 1 for yi in y]
    throw(ArgumentError("LaplaceGP labels must encode two classes as Bool, {0,1}, or {-1,+1}; got value set $(sort(vals))"))
end

"""
    update(g::LaplaceGP, X, y) -> LaplaceGP

Condition the classifier on new labelled inputs `(X, y)`. Because the Laplace MAP
is re-fit from scratch, the new data is appended to the stored history and the MAP
solve (`_laplace_fit`) is rerun over the full set.

Labels `y` may be `Bool`, `{0,1}` integers, or `{-1,+1}` integers; they are coerced
to `Bool` before fitting.
"""
function update(g::LaplaceGP, X::AbstractVector, y::AbstractVector)
    yb = _to_labels(y)
    _validate_obs(X, yb)
    xall = vcat(g.x, collect(X))
    yall = vcat(g.y, yb)
    a, W, L = _laplace_fit(g.prior, xall, yall)
    return LaplaceGP(g.prior, xall, yall, a, W, L)
end

"""
    _latent_moments(g::LaplaceGP, xs) -> (μ, σ²)

Posterior mean and marginal variance of the *latent* function at `xs` (R&W eqs.
3.21–3.24). The variance subtracts `‖v‖²` columnwise, where `v = L⁻¹(√W·cov(x, xs))`.
"""
function _latent_moments(g::LaplaceGP, xs)
    Ks = AbstractGPs.cov(g.prior, g.x, xs); sW = sqrt.(g.W)
    μ = AbstractGPs.mean(g.prior, xs) .+ Ks' * g.a
    v = g.L \ (sW .* Ks)                                      # whitened test cross-cov
    σ² = AbstractGPs.var(g.prior, xs) .- vec(sum(v .^ 2; dims = 1))
    return μ, σ²
end
Statistics.mean(g::LaplaceGP, xs::AbstractVector) = _hasdata(g) ? _latent_moments(g, xs)[1] : AbstractGPs.mean(g.prior, xs)
Statistics.var(g::LaplaceGP, xs::AbstractVector) = _hasdata(g) ? _latent_moments(g, xs)[2] : AbstractGPs.var(g.prior, xs)
mean_and_var(g::LaplaceGP, xs::AbstractVector) = _hasdata(g) ? _latent_moments(g, xs) : (AbstractGPs.mean(g.prior, xs), AbstractGPs.var(g.prior, xs))
function Statistics.cov(g::LaplaceGP, xs::AbstractVector, ys::AbstractVector)
    c = AbstractGPs.cov(g.prior, xs, ys)
    _hasdata(g) || return c
    sW = sqrt.(g.W)
    vx = g.L \ (sW .* AbstractGPs.cov(g.prior, g.x, xs)); vy = g.L \ (sW .* AbstractGPs.cov(g.prior, g.x, ys))
    return c .- vx' * vy
end
Statistics.cov(g::LaplaceGP, xs::AbstractVector) =
    Diagonal(_hasdata(g) ? _latent_moments(g, xs)[2] : AbstractGPs.var(g.prior, xs))
"""Posterior latent mean at a single input `u`, as a scalar; `>0` predicts the positive class."""
predmean(g::LaplaceGP, u) = mean(g, [u])[1]

# v1: the Laplace path does no hyperparameter refit, so `fit` returns the GP unchanged.
fit(g::LaplaceGP; kwargs...) = g
