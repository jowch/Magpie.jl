using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Random, Test
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
@testset "gpude: train! reduces loss; posterior_gps predmean matches field" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    truef(u) = -0.5u + sin(u)
    Z = [[x] for x in range(-3, 3; length = 10)]
    u0 = [2.5]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 10))
    target = Array(solve(ODEProblem((u, p, t) -> [truef(u[1])], u0, tspan), Tsit5(); saveat = ts))
    field = Magpie.ExactGPField(SqExponentialKernel(), Z; d = 1)
    loss0 = ext.make_loss(field, u0, tspan, ts, target)(field.v0)
    Magpie.train!(field, (ts, target); tspan, adam_iters = 50, maxiters = 50)  # small iters: mechanism gate, not full recovery
    lossT = ext.make_loss(field, u0, tspan, ts, target)(field.v0)
    @info "train!" loss0 lossT
    @test lossT < loss0                                   # training reduced the loss
    gps = Magpie.posterior_gps(field)   # PUBLIC path (not ext.posterior_gps) — guards the qualified-extension fix
    vopt = field.v0
    pf = vcat(vopt[1], vopt[2], vec(Magpie.solve_alpha(field, vopt[1], vopt[2], field.lognoise, Magpie.wmat(Magpie.FieldLayout(10, 1), vopt))))
    @test Magpie.predmean(gps[1], [0.5]) ≈ Magpie.gpfield(field, [0.5], pf)[1] rtol = 1.0e-8
end
