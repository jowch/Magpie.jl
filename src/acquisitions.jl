abstract type AcquisitionFunction end
abstract type MarginalAcquisition <: AcquisitionFunction end
struct Straddle{T<:Real} <: MarginalAcquisition; h::T; β::T; end
Straddle(; h::Real=0.0, β::Real=1.96) = Straddle(promote(float(h), float(β))...)
function (a::Straddle)(g, x)
    μ, v = predict(g, [x])
    return a.β * sqrt(v[1]) - abs(μ[1] - a.h)
end
