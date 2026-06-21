"""
    grad_predict(g::ExactGP, x) -> (μ∇, Σdiag, H)

Posterior gradient mean `μ∇` (length d), marginal gradient variances `Σdiag`
(length d, one per ∂f/∂xᵢ), and posterior-mean Hessian `H` (d×d) of an RBF
`ExactGP` at query `x`, observing f-values only. RBF + ZeroMean prior assumed.

Math (k(x,x')=exp(−‖x−x'‖²/2ℓ²), r=x−Xⱼ):
  ∂ᵢk(x,Xⱼ)        = −(rᵢ/ℓ²) k
  ∂²k/∂xᵢ∂xⱼ       = k(rᵢrⱼ/ℓ⁴ − δᵢⱼ/ℓ²)        (query Hessian)
  Var[∂ᵢf] prior   = 1/ℓ²,  posterior = 1/ℓ² − (∂ᵢk(x,X)) C⁻¹ (∂ᵢk(x,X))ᵀ
"""
function grad_predict(g::ExactGP, x::AbstractVector; hessian::Bool=true)
    d = length(x); ℓ = _lengthscale(g.prior.kernel)
    if isempty(g.x)                       # no data → prior
        return (zeros(d), fill(1/ℓ^2, d), hessian ? zeros(d, d) : nothing)
    end
    n = length(g.x)
    G = Matrix{Float64}(undef, d, n)      # Gᵢⱼ = ∂ᵢk(x, Xⱼ)
    Hsum = zeros(d, d); s = 0.0
    for j in 1:n
        Xj = g.x[j]; r = x .- Xj
        kj = g.prior.kernel(x, Xj)        # scalar RBF value (lengthscale baked in)
        @views G[:, j] .= .-(r ./ ℓ^2) .* kj
        if hessian                        # skip O(n·d²) Hessian work on the acquisition hot path
            Hsum .+= (g.α[j] * kj) .* (r * r')
            s += g.α[j] * kj
        end
    end
    μ∇ = G * g.α
    Σdiag = max.((1/ℓ^2) .- diag_Xt_invA_X(g.C, permutedims(G)), 0.0)   # permutedims → n×d
    H = hessian ? Matrix(Symmetric(Hsum ./ ℓ^4 .- (s/ℓ^2) .* Matrix(I, d, d))) : nothing
    return (μ∇, Σdiag, H)
end
