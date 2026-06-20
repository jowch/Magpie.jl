module AlphaGP

using LinearAlgebra, Statistics, Random
using AbstractGPs, KernelFunctions
using AbstractGPs: update_chol, Xt_invA_X, Xt_invA_Y, diag_Xt_invA_X
using ForwardDiff                      # loads DI's ForwardDiff extension (AutoForwardDiff)
import Statistics: mean, var, cov
import StatsBase: mean_and_var

include("spine.jl"); include("fit.jl"); include("laplace.jl")
include("acquisitions.jl"); include("maximize.jl"); include("loop.jl")

export ExactGP, update, predmean, predict, nlml
export LaplaceGP
export AcquisitionFunction, MarginalAcquisition, Straddle, RandStraddle, BinaryBALD, resample
export Box, Points, SobolPolish, grid_points, acquire
export ActiveLearner, observe!, fit!, run!, posterior_gp, queried_points, all_data

end
