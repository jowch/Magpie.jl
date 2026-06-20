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
