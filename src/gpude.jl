# Capability B core — pure (no SciML import). See ext/MagpieSciMLExt.jl for solve-touching code.

"""Trainable GP-UDE field: one shared kernel + fixed anchors `Z`, `d` independent outputs."""
struct ExactGPField{Tp,TZ}
    prior::Tp        # AbstractGPs.GP (mean + kernel); kernel hypers are overridden per-eval from pf
    Z::TZ            # Vector{Vector{Float64}} of anchors
    n::Int           # number of anchors
    d::Int           # output dimension
    lognoise::Float64     # FIXED jitter on anchor values (the field is never observed; identifiability mech. 2)
    v0::Vector{Float64}   # initial flat TRAINED params [logℓ, logσ, vec(w)] — lognoise is NOT trained here
end

"""Sparse variational GP as a full spine `AbstractGPModel` (per interface-design). Stores enough to
serve the whole AbstractGP contract via the whitened moments: `α = L_ZZ'\\μ` (predmean), plus the
inducing Cholesky `L_ZZ` and the variational factor `L_S` (var/cov — needed by SVGP PULL uncertainty).
mean/var/cov methods are defined in Task 7 via `svgp_moments`."""
struct SparseGP{Tp,TZ,Tα,TL,TS} <: AbstractGPModel
    prior::Tp; Z::TZ; α::Tα; L_ZZ::TL; L_S::TS
end

"Flat-Vector layout helper for Stage-1/2 trained params `[logℓ, logσ, vec(w)]` (lognoise is fixed on the field)."
struct FieldLayout; n::Int; d::Int; end

# Shooting + uncertainty selector types (named by users; methods live in the ext).
struct SingleShooting end
struct MultipleShooting; nsegments::Int; λ::Float64; λ0::Float64; end
MultipleShooting(; nsegments, λ=100.0, λ0=1e4) = MultipleShooting(nsegments, λ, λ0)
struct PULL end
struct Pathwise; n::Int; end
Pathwise(; n=128) = Pathwise(n)

# Stubs implemented in MagpieSciMLExt (require OrdinaryDiffEq + SciMLSensitivity).
# Declared in CORE so the ext can EXTEND them and `using Magpie: train!` resolves
# (an extension cannot inject new standalone names into the parent namespace).
function train! end
function propagate end
function posterior_gps end
function posterior_sparsegps end
train!(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
propagate(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
posterior_gps(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
posterior_sparsegps(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")

"""Multi-output SVGP field: ONE shared set of `M` inducing points `Z` (in state space), per-output
variational `(μ, L_S)`, `dout` independent outputs. Trained vector (verified shared-Z layout):
`[logℓ, logσ, vec(Z)(D·M), vec(μ)(M·dout), vec(L_S)(dout·nLS(M))]`; jitter is fixed on the field."""
struct SVGPField{Tp,TZ}
    prior::Tp; Z0::TZ; M::Int; dout::Int; D::Int; jitter::Float64; v0::Vector{Float64}
end
