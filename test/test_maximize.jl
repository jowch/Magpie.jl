using Magpie, AbstractGPs, KernelFunctions, Test
using Magpie: ExactGP, Straddle, Box, Points, acquire

@testset "acquire maximizes the acquisition" begin
    g = Magpie.update(ExactGP(with_lengthscale(SqExponentialKernel(), 0.4); noise = 1.0e-4), [[0.0], [1.0]], [0.0, 1.0])
    a = Straddle(h = 0.5); cands = [[x] for x in range(0, 1; length = 201)]

    # (a) acquire over Points returns the argmax
    xb = acquire(g, a; over = Points(cands))
    @test a(g, xb) ≈ maximum(a(g, c) for c in cands) rtol = 1.0e-10

    # (b) acquire over Box returns a point at least as good as a corner
    xb2 = acquire(g, a; over = Box([0.0], [1.0]))
    @test a(g, xb2) ≥ a(g, [0.0]) - 1.0e-6
end
