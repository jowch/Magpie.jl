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
    # buffer=20: FULL coherent recurrence including the cross-cov Dₙ term (the "past does matter" term)
    μs, Σs = ext.pull_propagate([gp], u0, ts; buffer=20)
    # buffer=0: Dₙ dropped — must UNDERESTIMATE Σ (the load-bearing canary)
    μs0, Σs0 = ext.pull_propagate([gp], u0, ts; buffer=0)
    β = only(var(gp, [0.0]))
    @assert β > 1e-4 "β vacuous — the oracle/canary would be meaningless"
    # PULL paper (arXiv:2211.11103) eq 21b — coherent linear-field flow for du/dt = a·u, Σ_0=0:
    #   Σ(t) = (β/a²)(1 − exp(a·t))²   (coherent/quadratic onset, NOT the white-noise (β/−2a)(1−e^{2at})).
    # ADVISORY only: eq 21b is exact for a CONSTANT-uncertainty linear field, but our GP's σ²_f(x) varies
    # in space (≈0 at anchors, larger between), so a single β can't match tightly — the quantitative gate
    # is the Task-10 PULL-vs-Pathwise Monte-Carlo cross-check. Here we keep robust sanity + the canary.
    oracle(t) = (β / a^2) * (1 - exp(a*t))^2
    @info "PULL eq-21b oracle (advisory)" Σend=Σs[end][1,1] oracle_end=oracle(ts[end]) β
    @test all(isfinite(Σ[1,1]) && Σ[1,1] ≥ 0 for Σ in Σs)              # finite, non-negative throughout
    @test Σs[end][1,1] > Σs[2][1,1] > 0                                 # uncertainty grows along the trajectory
    @test oracle(ts[end])/10 < Σs[end][1,1] < 10*oracle(ts[end])       # within an order of magnitude of eq 21b
    # LOAD-BEARING: dropping Dₙ (buffer=0) underestimates Σ vs the full coherent recurrence.
    @test Σs0[end][1,1] < Σs[end][1,1]
end
