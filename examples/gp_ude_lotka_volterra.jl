# # GP-UDE: Lotka-Volterra
#
# Learn the predator–prey vector field through the solver.
#
# A single trajectory over a short horizon `(0, 3)` is enough for single-shooting
# to recover the limit-cycle dynamics.  The `ExactGPField` is trained end-to-end
# with `train!` (ADAM warm-up → LBFGS polish), and the recovered posterior GPs
# are used to propagate uncertainty over a held-out horizon.

ENV["GKSwstype"] = "100"  ## GR headless
using Magpie, OrdinaryDiffEq, SciMLSensitivity, KernelFunctions, LinearAlgebra, Random
using Plots; gr()
Random.seed!(42)

# ## True system

function lv!(du, u, p, t)
    α, β, γ, δ = 1.5, 1.0, 3.0, 1.0
    du[1] = α*u[1] - β*u[1]*u[2]
    du[2] = δ*u[1]*u[2] - γ*u[2]
    nothing
end

u0 = [1.0, 1.0]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length=15))
target = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat=ts))

# ## Build the GP-UDE field

Z = kmeans_anchors(target, 12; rng=MersenneTwister(7))
field = ExactGPField(SqExponentialKernel(), Z; d=2)

# `train!` uses ADAM (1000 steps, lr=0.05) → LBFGS (200 iters) by default.
# `λ=1/(15*2)` is a weak log-ℓ prior; `s=0.5` is its standard deviation.
field, vopt = train!(field, (ts, target); tspan, maxiters=150, λ=1/(15*2), s=0.5)

# ## Posterior GPs and trajectory RMSE

gps = posterior_gps(field, vopt)

# Propagate uncertainty from u0 over the training horizon (PULL moment-matching).
μs, Σs = propagate(gps, u0, tspan; method=PULL(), ts=ts)
μmat = reduce(hcat, μs)   # 2 × |ts|

p1 = plot(ts, target[1,:], label="prey (true)", lw=2, c=:blue)
plot!(p1, ts, target[2,:], label="pred (true)", lw=2, c=:red)
plot!(p1, ts, μmat[1,:], label="prey (GP)", ls=:dash, lw=2, c=:blue)
plot!(p1, ts, μmat[2,:], label="pred (GP)", ls=:dash, lw=2, c=:red)
xlabel!(p1, "t"); ylabel!(p1, "population"); title!(p1, "LV: GP-UDE trajectory")
savefig(p1, "lv_trajectory.png")

# ## Anti-rot assertion (#src lines run on direct execution only)

using Test  #src
ext = Base.get_extension(Magpie, :MagpieSciMLExt)                                #src
dloss = ext.make_loss(field, Magpie.FieldLayout(field.n, field.d), u0, tspan, ts, target; λ=0.0, λσ=0.0) #src
## dloss is now a Gaussian NLL; recover the SSE = 2σ²·(NLL − (Nd/2)·log(2πσ²)) for the RMSE metric. #src
σ2 = exp(2*vopt[Magpie.NHYP]); Nd = 15*2                                          #src
sol_rmse = sqrt(2σ2*(dloss(vopt) - (Nd/2)*log(2π*σ2)) / Nd)                       #src
@info "LV sol_rmse = $sol_rmse"                                                  #src
@test sol_rmse < 0.3                                                             #src
