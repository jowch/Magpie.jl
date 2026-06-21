module MagpieSciMLExt

using Magpie
using Magpie: _chol, predmean, ExactGP, ExactGPField, SparseGP, FieldLayout, gpfield, solve_alpha
using OrdinaryDiffEq
using SciMLSensitivity
import SciMLBase
using LinearAlgebra
using KernelFunctions, AbstractGPs
import DifferentiationInterface as DI
import Mooncake
import Optimization
import OptimizationOptimJL: LBFGS

# MooncakeVJP is UNEXPORTED — bind once (SciMLSensitivityMooncakeExt auto-fires; Mooncake is a core hard dep).
const MOONCAKEVJP = SciMLSensitivity.MooncakeVJP()
const DEFAULT_SENSEALG = GaussAdjoint(autojacvec = MOONCAKEVJP)

# Stage-1 single-shooting loss. α recomputed in-loss (R1); Array(sol) extraction (R2).
function make_loss(field::ExactGPField, L::FieldLayout, u0, tspan, ts, X;
                   known_physics=(u,t)->zero(u), solver=Tsit5(), sensealg=DEFAULT_SENSEALG,
                   λ=1.0, logℓ_ref=0.0, s=0.5,             # logℓ prior
                   λσ=1.0, sσ=1.0)                          # weak logσ prior (breaks the ℓ–σ ridge)
    rhs!(du, u, pf, t) = (du .= known_physics(u, t); du .+= gpfield(field, u, pf); nothing)
    return function loss(v)
        h  = Magpie.hyp(L, v)
        α  = solve_alpha(field, h.logℓ, h.logσ, field.lognoise, Magpie.wmat(L, v))  # lognoise FIXED
        pf = vcat(h.logℓ, h.logσ, vec(α))
        sol = solve(ODEProblem(rhs!, u0, tspan, pf), solver; saveat=ts, sensealg)
        data = sum(abs2, Array(sol) .- X)              # R2: Array(sol), never sol[:,i]
        reg = λ*(h.logℓ - logℓ_ref)^2/(2s^2) + λσ*h.logσ^2/(2sσ^2)
        return data + reg
    end
end

# ---------------------------------------------------------------------------
# _init_vec: build initial optimisation vector from the field + shooting strategy
# ---------------------------------------------------------------------------

# SingleShooting: optimise field params only — just copy v0
_init_vec(field, u_data, t_data, ::Magpie.SingleShooting) = copy(field.v0)

# MultipleShooting: append per-segment initial nodes (seeded from data) to the field params
function _init_vec(field, u_data, t_data, ms::Magpie.MultipleShooting)
    idx = round.(Int, range(1, length(t_data); length=ms.nsegments+1))
    s0  = hcat([collect(u_data[:, idx[i]]) for i in 1:ms.nsegments]...)   # d × S nodes from data
    return vcat(copy(field.v0), vec(s0))
end

# ---------------------------------------------------------------------------
# build_loss: Stage-1 SingleShooting dispatch (Stage-2 MultipleShooting in Task 6)
# ---------------------------------------------------------------------------

# `kw...` forwards solver/sensealg/λ/logℓ_ref/s/λσ/sσ to make_loss.
function build_loss(field, L, u_data, t_data, tspan, ::Magpie.SingleShooting; kw...)
    u0 = u_data isa AbstractMatrix ? collect(u_data[:, 1]) : [u_data[1]]
    make_loss(field, L, u0, tspan, collect(t_data), u_data; kw...)
end

# ---------------------------------------------------------------------------
# train!: optimizer driver — Optimization.jl + LBFGS, outer Mooncake
# ---------------------------------------------------------------------------

function Magpie.train!(field::ExactGPField, (t_data, u_data);
                       tspan=(first(t_data), last(t_data)), known_physics=(u,t)->zero(u),
                       solver=Tsit5(), sensealg=DEFAULT_SENSEALG, shooting=Magpie.SingleShooting(),
                       ad=DI.AutoMooncake(; config=nothing), optimizer=LBFGS(), maxiters=200, kw...)
    L = FieldLayout(field.n, field.d)
    loss = build_loss(field, L, u_data, t_data, tspan, shooting; known_physics, solver, sensealg, kw...)
    v_init = _init_vec(field, u_data, t_data, shooting)
    optf = Optimization.OptimizationFunction((v, _p) -> loss(v), ad)
    sol  = Optimization.solve(Optimization.OptimizationProblem(optf, v_init), optimizer; maxiters)
    field.v0 .= sol.u[1:length(field.v0)]          # store FIELD prefix; drop s0 for MultipleShooting
    return field, sol.u
end

# ---------------------------------------------------------------------------
# posterior_gps: reconstruct d single-output ExactGPs from trained params
# ---------------------------------------------------------------------------

# Convenience overload using the stored trained params
posterior_gps(field::ExactGPField) = posterior_gps(field, field.v0)

function posterior_gps(field::ExactGPField, v)
    L = FieldLayout(field.n, field.d); h = Magpie.hyp(L, v)
    k = Magpie._kernel(h.logℓ, h.logσ)
    K = kernelmatrix(k, field.Z) + exp(field.lognoise) * I    # lognoise is on the field, not in v
    C = _chol(K)
    α = K \ Magpie.wmat(L, v)                      # (n×d) weights
    prior = AbstractGPs.GP(field.prior.mean, k)
    return [ExactGP(prior, field.Z, zeros(field.n), C, α[:, i], exp(field.lognoise)) for i in 1:field.d]
end

end # module
