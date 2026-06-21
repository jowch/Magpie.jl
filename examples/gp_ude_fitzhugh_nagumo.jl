# # GP-UDE: FitzHugh-Nagumo (UDE decomposition)
#
# Known linear dynamics + a GP that learns the unknown cubic nonlinearity.
#
# The FitzHugh-Nagumo system is:
#   v̇ = v - v³/3 - w + I_ext
#   ẇ = (v + a - b·w) / τ
#
# The `known_physics` argument carries the *known* linear part `[v - w + I, (v+a-bw)/τ]`,
# so the GP only needs to learn the residual cubic term `[-v³/3, 0]` — a much simpler
# regression target. This is the **UDE (Universal Differential Equation) decomposition**
# pattern: domain knowledge shrinks the GP's task.
#
# Short horizon `(0, 10)` for reliable single-shooting recovery.

ENV["GKSwstype"] = "100"  ## GR headless
using Magpie, OrdinaryDiffEq, SciMLSensitivity, KernelFunctions, LinearAlgebra, Random
using Plots; gr()
Random.seed!(3)

# ## True system

const FHN_a, FHN_b, FHN_τ, FHN_I = 0.7, 0.8, 12.5, 0.5

function fhn!(du, u, p, t)
    v, w = u[1], u[2]
    du[1] = v - v^3/3 - w + FHN_I
    du[2] = (v + FHN_a - FHN_b*w) / FHN_τ
    nothing
end

u0 = [-1.0, 1.0]; tspan = (0.0, 10.0); ts = collect(range(tspan...; length=25))
target = Array(solve(ODEProblem(fhn!, u0, tspan), Tsit5(); saveat=ts))

# ## UDE split — known linear part; GP learns only [-v³/3, 0]

# `known_physics(u, t)` returns the *known* part of the dynamics.
# The GP will learn what's left — ideally the cubic correction [-v³/3, 0].
fhn_known(u, t) = [u[1] - u[2] + FHN_I, (u[1] + FHN_a - FHN_b*u[2]) / FHN_τ]

Z = kmeans_anchors(target, 14; rng=MersenneTwister(3))
field = ExactGPField(SqExponentialKernel(), Z; d=2)

field, vopt = train!(field, (ts, target);
                     tspan, known_physics=fhn_known, maxiters=150, λ=1/(25*2))

# ## Posterior GPs

gps = posterior_gps(field, vopt)

# Visualize trajectory recovery
μs, Σs = propagate(gps, u0, tspan; method=PULL(), ts=ts)
μmat = reduce(hcat, μs)

# Add known_physics to PULL propagation mean for the full field
fhntot_mean(u) = fhn_known(u, 0.0) .+ [predmean(gps[i], u) for i in 1:2]

p1 = plot(ts, target[1,:], label="v (true)", lw=2, c=:blue)
plot!(p1, ts, target[2,:], label="w (true)", lw=2, c=:red)
xlabel!(p1, "t"); ylabel!(p1, "state"); title!(p1, "FHN: GP-UDE trajectory")
savefig(p1, "fhn_trajectory.png")

# ## Anti-rot assertion (#src lines run on direct execution only)

using Test  #src
ext = Base.get_extension(Magpie, :MagpieSciMLExt)                                              #src
dloss = ext.make_loss(field, Magpie.FieldLayout(field.n, field.d), u0, tspan, ts, target;      #src
                      known_physics=fhn_known, λ=0.0, λσ=0.0)                                  #src
sol_rmse = sqrt(dloss(vopt) / (25*2))                                                           #src
@info "FHN sol_rmse = $sol_rmse"                                                               #src
@test sol_rmse < 0.6                                                                            #src
