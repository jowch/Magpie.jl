using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Statistics, Test
using OrdinaryDiffEq, SciMLSensitivity
import ForwardDiff
using Magpie: ExactGP, update, predmean, PULL, Pathwise

@testset "PULL: predmean input-Jacobian matches FD" begin
    k = Magpie._kernel(0.0, 0.0); Z=[[x] for x in range(-2,2;length=6)]
    gp = update(ExactGP(k; noise=1e-6), Z, sinpi.(first.(Z)))
    J = ForwardDiff.gradient(u -> predmean(gp, u), [0.3])
    fd = (predmean(gp,[0.3+1e-6]) - predmean(gp,[0.3-1e-6]))/2e-6
    @test only(J) ≈ fd rtol=1e-5
end

@testset "PULL: linear oracle + Dₙ=0 canary (the load-bearing assertion)" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    # linear field f(u)=a·u as a GP; noise≈1e-3 + ~12 anchors so β=var(gp,·) is O(1e-2) (NON-vacuous).
    a = -0.6; Z=[[x] for x in range(-3,3;length=12)]
    gp = update(ExactGP(Magpie._kernel(log(0.8),0.0); noise=1e-3), Z, a .* first.(Z))
    u0=[1.0]; ts=collect(range(0,2.0;length=11))
    # buffer=20: full recurrence including cross-cov Dₙ
    μs, Σs = ext.pull_propagate([gp], u0, ts; buffer=20)
    # buffer=0: Dₙ=0 — only the diagonal injection h·Vₙ. Oracle tested here (Dₙ=0 ⇒ IID).
    μs0, Σs0 = ext.pull_propagate([gp], u0, ts; buffer=0)
    β = only(var(gp, [0.0]))
    @assert β > 1e-4 "β vacuous — the oracle/canary would be meaningless"
    # CORRECT continuous-time form for du/dt = a·u with IID injection β (per unit time):
    # Σ(t) = (β/(-2a))(1 - exp(2at)). Tested against buffer=0 where Dₙ=0 (IID assumption holds).
    oracle(t) = (β / (-2a)) * (1 - exp(2a*t))
    err = maximum(abs(Σs0[i][1,1] - oracle(ts[i]))/max(oracle(ts[i]),1e-8) for i in 2:length(ts))
    @info "PULL oracle" err β
    @test err < 0.25                       # SANITY: V_n varies along trajectory so β is approximate.
    # LOAD-BEARING: dropping Dₙ underestimates Σ (one-sided; robust to oracle-formula error & identifiability).
    @test Σs0[end][1,1] < Σs[end][1,1]
end
