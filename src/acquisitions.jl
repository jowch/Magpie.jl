using StatsFuns: normcdf

abstract type AcquisitionFunction end
abstract type MarginalAcquisition <: AcquisitionFunction end
struct Straddle{T<:Real} <: MarginalAcquisition; h::T; β::T; end
Straddle(; h::Real=0.0, β::Real=1.96) = Straddle(promote(float(h), float(β))...)
function (a::Straddle)(g, x)
    μ, v = predict(g, [x])
    return a.β * sqrt(v[1]) - abs(μ[1] - a.h)
end

struct RandStraddle{T<:Real,R} <: MarginalAcquisition; h::T; sβ::T; rng::R; end
RandStraddle(; h::Real=0.0, rng=Random.default_rng()) = RandStraddle(float(h), sqrt(-2*log(rand(rng))), rng)
function (a::RandStraddle)(g, x)
    μ, v = predict(g, [x]); σ = sqrt(v[1])
    return max(min(μ[1] + a.sβ*σ - a.h, a.h - (μ[1] - a.sβ*σ)), zero(σ))
end
resample(a::AcquisitionFunction) = a
resample(a::RandStraddle) = RandStraddle(a.h, sqrt(-2*log(rand(a.rng))), a.rng)

struct BinaryBALD <: AcquisitionFunction end
const _C_BALD = sqrt(π*log(2)/2)
const _λ_LOGIT = sqrt(π/8)                            # logistic→probit correction
# binary entropy in BITS (log2) — the Houlsby constant C=√(π·ln2/2) is calibrated for
# bits, so the entropy term must match (nats would make BALD spuriously negative).
_hbin(p) = (q = clamp(p, eps(), 1-eps()); -q*log2(q) - (1-q)*log2(1-q))
function (::BinaryBALD)(g, x)
    μ, v = predict(g, [x]); z = _λ_LOGIT * μ[1]
    # Houlsby closed form: H[y|x,D] − E_f[H[y|x,f]] ≈ h_b(Φ(z/√(v+1))) − C/√(v+C²)·exp(−z²/(2(v+C²))).
    # BALD is a mutual information (≥0); clamp the ~1e-3 negative dip of the approximation at high confidence.
    bald = _hbin(normcdf(z / sqrt(v[1] + 1))) - _C_BALD / sqrt(v[1] + _C_BALD^2) * exp(-z^2 / (2*(v[1] + _C_BALD^2)))
    return max(bald, zero(bald))
end
# MulticlassBALD is DEFERRED (needs AugmentedGPLikelihoods.jl) — not in this plan.
