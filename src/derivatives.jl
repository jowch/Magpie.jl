"""
    grad_predict(g::ExactGP, x; hessian=true) -> (μ∇, Σdiag, H)

Posterior gradient mean `μ∇` (length d), marginal gradient variances `Σdiag`
(length d, one per ∂f/∂xᵢ), and posterior-mean Hessian `H` (d×d) of an `ExactGP`
at query `x`, observing f-values only.

Kernel-generic: `μ∇` and `H` are AD (`ForwardDiff`) of the posterior mean
[`predmean`](@ref), so they need no hand-derived per-kernel algebra and pick up the
kernel output scale σ_f² for free. Only the prior gradient-variance term needs
kernel-specific care (see [`_prior_grad_var`](@ref)).

Math: with the posterior mean `μ(x)=m(x)+k(x,X)α`,
  `μ∇  = ∇ₓ μ(x)`,  `H = ∇²ₓ μ(x)`,
  `Var[∂ᵢf] = Var_prior[∂ᵢf] − (∂ᵢk(x,X)) C⁻¹ (∂ᵢk(x,X))ᵀ`.

# ponytail: AD path, kernel-general. Slower than hand-rolled RBF blocks (n kernel-grad
# calls for the variance, a d-dual Hessian only at classification time), but the thesis is
# to bridge KernelFunctions + AD, not hand-code derivatives per kernel. The acquisition
# uses `hessian=false`; the AD Hessian fires only at the handful of extracted candidates.
"""
function grad_predict(g::ExactGP, x::AbstractVector; hessian::Bool = true)
    g.d == 1 ||
        throw(ArgumentError("grad_predict is single-output (d=1); got d=$(g.d). Multi-output derivatives are not supported."))
    d = length(x); k = g.prior.kernel
    pv = _prior_grad_var(k, x)                                       # prior Var[∂ᵢf], length d
    # No data: gradient/Hessian are those of the prior mean (∇m, ∇²m) — zero for ZeroMean.
    _hasdata(g) || return (
        ForwardDiff.gradient(z -> predmean(g, z), x), pv,
        hessian ? ForwardDiff.hessian(z -> predmean(g, z), x) : nothing,
    )
    μ∇ = ForwardDiff.gradient(z -> predmean(g, z), x)
    G = reduce(hcat, (ForwardDiff.gradient(z -> k(z, Xj), x) for Xj in g.x))  # d×n: ∂ᵢk(x,Xⱼ)
    Σdiag = max.(pv .- diag_Xt_invA_X(g.C, permutedims(G)), 0.0)     # permutedims → n×d for helper
    H = hessian ? ForwardDiff.hessian(z -> predmean(g, z), x) : nothing
    return (μ∇, Σdiag, H)
end

"""
    _prior_grad_var(k, x) -> Vector

Per-component prior gradient variance `Var_prior[∂ᵢf(x)] = −∂²ₓᵢ k(x,x')|_{x'=x}`, i.e.
minus the diagonal of the query-Hessian of `k` at coincidence. Computed generically by AD
for kernels smooth at `r=0` (squared-distance kernels: SqExponential, RationalQuadratic, …).
Norm-distance kernels (Matérn, Exponential) are non-smooth at `r=0` so AD returns `NaN`
there — fall back to the analytic per-family constant via [`_prior_grad_var_const`](@ref).
"""
function _prior_grad_var(k, x)
    pv = -diag(ForwardDiff.hessian(z -> k(z, x), x))
    return all(isfinite, pv) ? pv : fill(_prior_grad_var_const(k), length(x))
end

# Analytic prior gradient variance for norm-singular kernels where AD NaNs at r=0.
# Var[∂ᵢf] = -2σ²·ψ'(0)/ℓ², ψ the kernel as a function of squared distance: c=5/3 (Matérn-5/2),
# c=3 (Matérn-3/2). Verified by central differences of the kernel. Add families here as needed.
function _prior_grad_var_const(k)
    bk = _basekernel(k); ℓ = _lengthscale(k); σ² = _outputscale(k)
    bk isa Matern52Kernel && return 5σ² / (3 * ℓ^2)
    bk isa Matern32Kernel && return 3σ² / ℓ^2
    error(
        "grad_predict: prior gradient variance is NaN under AD for $(typeof(bk)) " *
            "(norm-singular at r=0) and no analytic constant is registered; add one to _prior_grad_var_const"
    )
end
