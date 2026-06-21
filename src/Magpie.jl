"""
    Magpie

Composable Gaussian processes for active learning and dynamics, built on
[AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) and
[KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl).

A GP here is a differentiable, uncertainty-aware component: it composes with
autodiff (Mooncake), SciML, and the rest of the Julia ecosystem rather than
reinventing the GP core.

Two capabilities share one incrementally-updated GP spine:

  - **Active learning** — a `fit → acquire → observe → update` loop with level-set
    and classification-boundary acquisitions ([`Straddle`](@ref), [`BinaryBALD`](@ref)).
  - **GP-in-SciML** *(next milestone)* — a GP as the right-hand side of an ODE.

The extension point is [`AbstractGPModel`](@ref); the bundled implementations are
[`ExactGP`](@ref) (exact regression) and [`LaplaceGP`](@ref) (binary classification).
"""
module Magpie

using LinearAlgebra, Statistics, Random
using AbstractGPs, KernelFunctions
using AbstractGPs: update_chol, Xt_invA_X, Xt_invA_Y, diag_Xt_invA_X
using ForwardDiff                      # loads DI's ForwardDiff extension (AutoForwardDiff)
import Statistics: mean, var, cov
import StatsBase: mean_and_var

include("spine.jl"); include("fit.jl"); include("laplace.jl")
include("derivatives.jl"); include("acquisitions.jl"); include("maximize.jl"); include("loop.jl")

export AbstractGPModel, ExactGP, update, predmean, predict, nlml, grad_predict
export LaplaceGP
export AcquisitionFunction, MarginalAcquisition, Straddle, RandStraddle, BinaryBALD, resample
export Box, Points, SobolPolish, grid_points, acquire
export ActiveLearner, observe!, fit!, run!, posterior_gp, queried_points, all_data

end
