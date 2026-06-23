using Test, Magpie, LinearAlgebra
using OrdinaryDiffEq, SciMLSensitivity
using KernelFunctions, AbstractGPs

ext = Base.get_extension(Magpie, :MagpieSciMLExt)

@testset "divergence guard returns finite sentinel" begin
    # A field whose params make pf contain NaN ⇒ the ODE RHS returns NaN at every step ⇒
    # Tsit5 detects NaN in the initial-dt estimate and exits immediately (zero saved points)
    # ⇒ Array(sol) has wrong shape ⇒ size(A) != size(X) ⇒ guard fires → finite sentinel.
    Z = [[x] for x in range(-1, 1; length = 4)]
    field = ExactGPField(Magpie._kernel(0.0, 0.0), Z; d = 1)
    # Weights = NaN ⇒ α = K_ZZ⁻¹·NaN = NaN ⇒ pf tail = NaN ⇒ RHS = NaN every step ⇒
    # Tsit5 exits immediately (zero saved points) ⇒ Array(sol) wrong shape ⇒ guard fires.
    # logℓ/logσ stay 0 so regularizer = 0 (no NaN poisoning of the total loss).
    v = copy(field.v0)
    v[(Magpie.NHYP + 1):end] .= NaN          # field weights = NaN → propagates to α → pf → RHS = NaN
    ts = collect(range(0.0, 5.0; length = 20))
    X = zeros(1, length(ts))                  # data is irrelevant; we only check the sentinel
    loss = ext.field_loss(field, Magpie.SingleShooting(), [(ts, X)]; u0 = [0.0], tspan = (0.0, 5.0))
    L = loss(v)
    @test isfinite(L)                         # guard converts blow-up to a finite sentinel
    @test L ≥ 1.0e6                            # the sentinel value (plus regularizer)
end
