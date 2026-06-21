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

# methods added in later tasks: ms_loss, build_loss, svgp_elbo_loss, train!, propagate, pull_propagate

end # module
