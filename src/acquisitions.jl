using StatsFuns: normcdf, normlogcdf

"""
    AcquisitionFunction

Supertype for active-learning acquisitions. An acquisition is callable as `a(g, x)`,
scoring a candidate input `x` under the current GP `g`; the loop queries where the
score is largest (see [`acquire`](@ref)).
"""
abstract type AcquisitionFunction end

"""
    MarginalAcquisition <: AcquisitionFunction

Acquisitions that depend only on the GP's marginal posterior at `x` (its mean and
variance), so each candidate can be scored independently.
"""
abstract type MarginalAcquisition <: AcquisitionFunction end

@doc raw"""
    Straddle{T} <: MarginalAcquisition

Level-set "straddle" acquisition (Bryan et al. 2005):

```math
a(x) = \beta\,\sigma(x) - |\mu(x) - h|.
```

It is large where the posterior is uncertain (``\sigma`` high) *and* close to the
target level ``h``, driving queries onto the level set ``\{x : f(x) = h\}``.

# Constructor

    Straddle(; h=0.0, β=1.96)

`h` is the target level and `β` the exploration weight (the default ≈ a 95% band).
"""
struct Straddle{T <: Real} <: MarginalAcquisition
    h::T; β::T
end
Straddle(; h::Real = 0.0, β::Real = 1.96) = Straddle(promote(float(h), float(β))...)
function (a::Straddle)(g, x)
    μ, v = predict(g, [x])
    return a.β * sqrt(v[1]) - abs(μ[1] - a.h)
end

"""
    RandStraddle{T,R} <: MarginalAcquisition

Randomized straddle (Inatsu et al. 2024): like [`Straddle`](@ref) but the band width
`β` is redrawn each round as `√(−2 log u)`, `u ~ U(0,1)`, giving a theoretically
grounded exploration schedule. Call [`resample`](@ref) between rounds to redraw.

# Constructor

    RandStraddle(; h=0.0, rng=Random.default_rng())

`h` is the target level; `rng` seeds the band-width draw. The field `sβ` stores the
current `√β`.
"""
struct RandStraddle{T <: Real, R} <: MarginalAcquisition
    h::T; sβ::T; rng::R
end
RandStraddle(; h::Real = 0.0, rng = Random.default_rng()) = RandStraddle(float(h), sqrt(-2 * log(rand(rng))), rng)
function (a::RandStraddle)(g, x)
    μ, v = predict(g, [x]); σ = sqrt(v[1])
    return max(min(μ[1] + a.sβ * σ - a.h, a.h - (μ[1] - a.sβ * σ)), zero(σ))
end

"""
    resample(a::AcquisitionFunction[, rng]) -> AcquisitionFunction

Return a copy of `a` with any per-round randomness redrawn. Stochastic acquisitions
like [`RandStraddle`](@ref) draw a fresh band width; deterministic ones return `a`.
The two-argument form `resample(a, rng)` injects an explicit RNG (primary method);
the single-argument form `resample(a)` uses the RNG stored in `a` (backward-compatible
delegate).
"""
resample(a::AcquisitionFunction, rng) = a
resample(a::RandStraddle, rng) = RandStraddle(a.h, sqrt(-2 * log(rand(rng))), rng)
resample(a::AcquisitionFunction) = a
resample(a::RandStraddle) = resample(a, a.rng)

@doc raw"""
    GradStraddle(; β=1.96) <: AcquisitionFunction

Vector-zero straddle: a component-wise Straddle on the GP gradient, summed over
components. Scores high where every `∂f/∂xᵢ` is near zero AND uncertain, so the
loop samples toward the zeros of ∇f (the critical points of f).

```math
\text{score}(x) = \sum_i \left[ \beta\,\sqrt{\mathrm{Var}[\partial_i f(x)]} - |\mathbb{E}[\partial_i f(x)]| \right]
```

Sum (not min) over components: `min` is dominated by the most-resolved component
and starves half-resolved critical points; the sum keeps explore/exploit tension
on every component.
"""
struct GradStraddle{T <: Real} <: AcquisitionFunction
    β::T
end
GradStraddle(; β::Real = 1.96) = GradStraddle(float(β))
function (a::GradStraddle)(g, x)
    μ∇, Σdiag, _ = grad_predict(g, x; hessian = false)   # acquisition needs no Hessian
    return sum(a.β * sqrt(Σdiag[i]) - abs(μ∇[i]) for i in eachindex(μ∇))
end

@doc raw"""
    RandGradStraddle{R} <: AcquisitionFunction

Randomized [`GradStraddle`](@ref): the band width is redrawn each round as
`√(−2 log u)`, `u ~ U(0,1)` (the [`RandStraddle`](@ref) schedule). Randomizing the band
breaks the deterministic-argmax fixation that mode-collapses the plain `GradStraddle` on
landscapes with large gradient-magnitude variation — there the constant `−|μ∇|` term pins
the maximizer to one region, so a single-scale loop covers it but a steep multiscale one
does not. Call [`resample`](@ref) between rounds to redraw.
"""
struct RandGradStraddle{R} <: AcquisitionFunction
    sβ::Float64; rng::R
end
RandGradStraddle(; rng = Random.default_rng()) = RandGradStraddle(sqrt(-2 * log(rand(rng))), rng)
function (a::RandGradStraddle)(g, x)
    μ∇, Σdiag, _ = grad_predict(g, x; hessian = false)
    return sum(a.sβ * sqrt(Σdiag[i]) - abs(μ∇[i]) for i in eachindex(μ∇))
end
resample(a::RandGradStraddle, rng) = RandGradStraddle(sqrt(-2 * log(rand(rng))), rng)
resample(a::RandGradStraddle) = resample(a, a.rng)

@doc raw"""
    LocalPenalization(base, pts=[]; c=0.5, s=0.15) <: AcquisitionFunction

Diversity wrapper around any `base` acquisition: discourages querying within ``\sim c\ell``
of any point in `pts` (the observation history; ``\ell`` is the GP's fitted lengthscale).
Each point adds a soft penalty to the score,

```math
\sum_j \log \Phi\!\left(\frac{\lVert x - x_j\rVert - c\,\ell}{s\,\ell}\right),
```

which is ``\approx 0`` far from `xⱼ` and strongly negative inside the radius. This breaks the
deterministic-argmax **resampling collapse** — stacking queries at one point never resolves
the gradient there, so a plain straddle can sample it indefinitely and starve coverage.

In the active loop, pass the learner's **live** history so the penalty tracks new
observations with no loop changes (`observe!` mutates `al.Xs` in place):

```julia
al.acq = LocalPenalization(al.acq, al.Xs)
```

Adapts Local Penalization (González et al. 2016, *Batch Bayesian Optimization via Local
Penalization*) to gradient-zero / level-set search: the soft ``\Phi`` envelope is the same,
but the radius is the lengthscale ``c\ell`` (a redundancy scale) rather than LP's
optimum-seeking ``(M-\mu)/L``, which assumes a scalar being maximized — see the package
notes for why that radius does not transfer to this objective.
"""
struct LocalPenalization{A <: AcquisitionFunction, P} <: AcquisitionFunction
    base::A; pts::P; c::Float64; s::Float64
end
LocalPenalization(base::AcquisitionFunction, pts = Any[]; c::Real = 0.5, s::Real = 0.15) =
    LocalPenalization(base, pts, float(c), float(s))
function (a::LocalPenalization)(g, x)
    isempty(a.pts) && return a.base(g, x)
    ℓ = _lengthscale(g.prior.kernel); r = a.c * ℓ; sw = a.s * ℓ
    return a.base(g, x) + sum(normlogcdf((norm(x .- p) - r) / sw) for p in a.pts)
end
resample(a::LocalPenalization, rng) = LocalPenalization(resample(a.base, rng), a.pts, a.c, a.s)
resample(a::LocalPenalization) = LocalPenalization(resample(a.base), a.pts, a.c, a.s)

@doc raw"""
    BinaryBALD <: AcquisitionFunction

Bayesian Active Learning by Disagreement for binary classification (Houlsby et al.
2011): the mutual information between the label ``y`` and the latent function ``f``,

```math
\mathbb{I}[y; f \mid x, \mathcal{D}]
    = \mathrm{H}[y \mid x, \mathcal{D}] - \mathbb{E}_{f}\,\mathrm{H}[y \mid f],
```

computed in *bits*. Large where the GP is uncertain about the class probability,
driving queries that are most informative about the decision boundary. Pair with a
[`LaplaceGP`](@ref).
"""
struct BinaryBALD <: AcquisitionFunction end

# Houlsby constant C = √(π·ln2/2), calibrated so the closed-form BALD is in bits.
const _C_BALD = sqrt(π * log(2) / 2)
# Correction factor for the logistic→probit approximation σ(f) ≈ Φ(λf).
const _λ_LOGIT = sqrt(π / 8)

# Binary entropy in BITS (log2). Bits (not nats) are required: the Houlsby constant
# C is calibrated for bits, so a nats entropy term would make BALD spuriously negative.
_hbin(p) = (q = clamp(p, eps(), 1 - eps()); -q * log2(q) - (1 - q) * log2(1 - q))

function (::BinaryBALD)(g, x)
    μ, v = predict(g, [x])
    # Logistic→probit correction σ(f)≈Φ(λf): with f~N(μ,v), λf~N(λμ, λ²v) — BOTH moments
    # scale by λ. (Scaling only μ overestimates the true bits-BALD by ~2–2.5×; verified vs quadrature.)
    z = _λ_LOGIT * μ[1]; s² = _λ_LOGIT^2 * v[1]
    # Houlsby closed form: H[y|x,D] − E_f[H[y|x,f]] ≈ h_b(Φ(z/√(s²+1))) − C/√(s²+C²)·exp(−z²/(2(s²+C²))).
    # Calibrated bits-BALD to ~Houlsby approximation error (~10%); BALD is a mutual information (≥0),
    # so clamp the small negative dip the approximation can produce at high confidence.
    bald = _hbin(normcdf(z / sqrt(s² + 1))) - _C_BALD / sqrt(s² + _C_BALD^2) * exp(-z^2 / (2 * (s² + _C_BALD^2)))
    return max(bald, zero(bald))
end
# MulticlassBALD is DEFERRED (needs AugmentedGPLikelihoods.jl) — not in this plan.
