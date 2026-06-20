using Test
@testset "Magpie" begin
    include("test_spine.jl"); include("test_fit.jl"); include("test_ad.jl")
    include("test_acquisitions.jl"); include("test_laplace.jl")
    include("test_maximize.jl"); include("test_loop.jl"); include("test_exemplar_A1.jl")
    include("test_exemplar_A2.jl")
end
