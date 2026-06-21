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
include("acquisitions.jl"); include("maximize.jl"); include("loop.jl")
include("gpude.jl")

export AbstractGPModel, ExactGP, update, predmean, predict, nlml
export LaplaceGP
export AcquisitionFunction, MarginalAcquisition, Straddle, RandStraddle, BinaryBALD, resample
export Box, Points, SobolPolish, grid_points, acquire
export ActiveLearner, observe!, fit!, run!, posterior_gp, queried_points, all_data
export GPField, CompositeField, ExactGPField, SparseGP, SVGPField, FieldLayout, gpfield, solve_alpha, train!, propagate
export unpack, regularizer, posterior, posterior_gps, posterior_sparsegps
export SingleShooting, MultipleShooting, PULL, Pathwise, DecoupledGPSample, kmeans_anchors
export build_decoupled_sample
export svgp_kl, nLS, unpack_LS, L_ZZ_factor, svgp_moments

end
