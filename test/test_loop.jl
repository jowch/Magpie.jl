using AlphaGP, AbstractGPs, KernelFunctions, Test
using AlphaGP: ExactGP, Straddle, ActiveLearner, observe!, acquire, posterior_gp, all_data, Box

@testset "ActiveLearner loop == batch" begin
    al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(),0.5); noise=1e-4), Straddle(h=0.0))
    f(x) = sum(x)^2 - 1
    X = [[x] for x in range(-1,1;length=6)]
    for x in X; observe!(al, x, f(x)); end
    g_batch = AlphaGP.update(ExactGP(with_lengthscale(SqExponentialKernel(),0.5); noise=1e-4), X, f.(X))
    @test mean(posterior_gp(al), [[0.3],[-0.4]]) ≈ mean(g_batch, [[0.3],[-0.4]]) rtol=1e-9
    @test length(first(all_data(al))) == 6
    @test acquire(al; over=Box([-1.0],[1.0])) isa Vector
    @test_throws ErrorException acquire(al; over=Box([-1.0],[1.0]), q=3)
end
