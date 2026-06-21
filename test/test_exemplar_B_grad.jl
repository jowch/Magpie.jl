using Magpie, KernelFunctions, AbstractGPs, LinearAlgebra, Test
using Magpie: ExactGPField, FieldLayout
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake
using FiniteDifferences

@testset "B-grad: through-solver gradient matches FD, ∂loss/∂w live" begin
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    truef(u) = -0.5u + sin(u)
    Z = [[x] for x in range(-3, 3; length=10)]
    u0 = [2.5]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length=10))
    target = Array(solve(ODEProblem((u,p,t)->[truef(u[1])], u0, tspan), Tsit5(); saveat=ts))
    field = ExactGPField(SqExponentialKernel(), Z; d=1)
    L = FieldLayout(10, 1)
    v0 = vcat(log(1.3), 0.0, log(0.1), zeros(10))   # [logℓ, logσ, logσ_obs, vec(w)]; w=0 off-optimum so loss is non-flat in w
    loss = ext.make_loss(field, L, u0, tspan, ts, target)
    g_mc = DI.gradient(loss, DI.AutoMooncake(; config=nothing), v0)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), loss, v0)[1]
    relerr = norm(g_mc .- g_fd) / max(norm(g_fd), eps())
    @info "B-grad" relerr wnorm=norm(g_fd[Magpie.NHYP+1:end]) dσobs=abs(g_fd[Magpie.NHYP])
    @test relerr < 5e-3                          # Spike 1 saw ~1.3e-4
    @test norm(g_fd[Magpie.NHYP+1:end]) > 1e-3   # R3: ∂loss/∂w live (w-block starts after the hyper prefix)
    @test abs(g_fd[Magpie.NHYP]) > 1e-3          # R3 (Phase 2): the σ_obs slot is live
end
