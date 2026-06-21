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

# ---------------------------------------------------------------------------
# Pure field core — no SciML import.
# ---------------------------------------------------------------------------

"""
    ExactGPField(kernel, Z; d, mean, logℓ0, logσ0, lognoise)

Convenience constructor. Builds the zero-mean prior and default flat params
`v0 = [logℓ0, logσ0, zeros(n*d)]` (lognoise is NOT a trained slot).
"""
# lognoise=log(1e-2): the fixed anchor-value jitter. 1e-2 (not 1e-4) keeps K_ZZ well-conditioned so the
# Mooncake Cholesky-solve BACKWARD (potrs) doesn't hit SingularException when the optimizer explores
# long lengthscales / near-duplicate anchors — `_chol(check=false)` only guards the forward. Verified:
# 1e-2 removes the exception AND recovers LV better (sol_rmse 0.04 vs the fragile 1e-4).
function ExactGPField(kernel::Kernel, Z; d::Int=1, mean=AbstractGPs.ZeroMean(),
                      logℓ0=0.0, logσ0=0.0, lognoise=log(1e-2))
    n = length(Z)
    v0 = vcat(logℓ0, logσ0, zeros(n*d))      # lognoise is NOT a trained slot
    ExactGPField(AbstractGPs.GP(mean, kernel), collect(Z), n, d, Float64(lognoise), v0)
end

"""Output-scaled squared-exponential kernel: `exp(2logσ) * SE(exp(logℓ))`."""
_kernel(logℓ, logσ) = exp(2logσ) * with_lengthscale(SqExponentialKernel(), exp(logℓ))

# Layout accessors — trained vector is [logℓ, logσ, vec(w)]; lognoise lives on the field.
nw(L::FieldLayout) = L.n * L.d
hyp(L::FieldLayout, v) = (logℓ=v[1], logσ=v[2])
wmat(L::FieldLayout, v) = reshape(v[3:2+nw(L)], L.n, L.d)

"""
    solve_alpha(field, logℓ, logσ, lognoise, w) -> Matrix{n×d}

`α = (K_ZZ + σ_n² I)⁻¹ w` via a single shared Cholesky through `_chol`.
Call INSIDE the loss so `∂α/∂θ` stays alive through autodiff.
"""
function solve_alpha(field::ExactGPField, logℓ, logσ, lognoise, w)
    k = _kernel(logℓ, logσ)
    # RELATIVE jitter exp(lognoise)·σ² (σ²=exp(2logσ) is the kernel diagonal). Scaling with σ² bounds
    # cond(K_ZZ) ≈ 1 + n/exp(lognoise) regardless of how far logσ/logℓ drift — an ABSOLUTE jitter goes
    # negligible once logσ grows, and the Mooncake Cholesky-solve BACKWARD then throws SingularException.
    K = kernelmatrix(k, field.Z) + exp(lognoise + 2*logσ) * I
    return _chol(K) \ w
end

"""
    gpfield(field, u, pf) -> Vector{d}

Forward field callable. `pf = [logℓ, logσ, vec(α)...]`; `Z` is fixed (closed over `field`).
Returns `kuZ' * α` as a length-`d` vector.
"""
function gpfield(field::ExactGPField, u, pf)
    k   = _kernel(pf[1], pf[2])
    kuZ = [k(u, z) for z in field.Z]
    α   = reshape(@view(pf[3:end]), field.n, field.d)
    return vec(kuZ' * α)
end

"""
    kmeans_anchors(X, k; iters, rng) -> Vector{Vector{Float64}}

Tiny pure-Julia Lloyd's k-means on COLUMNS of `X` (each column = one d-dim state).
Returns `k` cluster centres as `Vector{Vector{Float64}}`.
"""
function kmeans_anchors(X::AbstractMatrix, k::Int; iters::Int=50, rng=Random.default_rng())
    d, N = size(X); @assert k <= N
    C = [Vector{Float64}(X[:, j]) for j in randperm(rng, N)[1:k]]
    assign = zeros(Int, N)
    for _ in 1:iters
        for i in 1:N
            best, bd = 1, Inf
            for c in 1:k
                dist = 0.0
                @inbounds for r in 1:d; dist += (X[r, i] - C[c][r])^2; end
                dist < bd && (bd = dist; best = c)
            end
            assign[i] = best
        end
        for c in 1:k
            m = findall(==(c), assign); isempty(m) && continue
            C[c] = vec(sum(@view(X[:, m]); dims=2)) ./ length(m)
        end
    end
    return C
end
