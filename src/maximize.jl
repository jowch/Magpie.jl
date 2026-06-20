using Optimization, OptimizationOptimJL, DifferentiationInterface
import Sobol

struct Box{T}; lb::T; ub::T; end
struct Candidates{M}; X::M; end

abstract type AcqMaximizer end
struct SobolPolish{A} <: AcqMaximizer; n_raw::Int; n_restarts::Int; ad::A; end
SobolPolish(; n_raw=2048, n_restarts=8, ad=AutoForwardDiff()) = SobolPolish(n_raw, n_restarts, ad)

function grid_candidates(box::Box; per_axis::Int=50)
    axes = [range(box.lb[i], box.ub[i]; length=per_axis) for i in eachindex(box.lb)]
    return [collect(p) for p in Iterators.product(axes...)] |> vec
end

default_for(::Candidates) = nothing
default_for(b::Box) = length(b.lb) ≤ 2 ? Val(:grid) : SobolPolish()

acquire(g, a; over, maximizer=default_for(over)) = _acquire(g, a, over, maximizer)

_argmax_over(g, a, X) = X[argmax([a(g, x) for x in X])]
_acquire(g, a, c::Candidates, _) = _argmax_over(g, a, c.X)
_acquire(g, a, b::Box, ::Val{:grid}) = _argmax_over(g, a, grid_candidates(b))

function _acquire(g, a, b::Box, m::SobolPolish)
    seq = Sobol.SobolSeq(b.lb, b.ub)
    raw = [Sobol.next!(seq) for _ in 1:m.n_raw]
    # rank raw candidates by descending acquisition value, pick top n_restarts as polish starts
    starts = sort(raw; by=x -> -a(g, x))[1:min(m.n_restarts, length(raw))]
    best = _argmax_over(g, a, raw); bestv = a(g, best)
    for x0 in starts
        prob = OptimizationProblem(
            OptimizationFunction((x, _) -> -a(g, x), m.ad),
            collect(x0);
            lb=b.lb, ub=b.ub,
        )
        sol = solve(prob, Fminbox(LBFGS()))
        v = -sol.objective
        if v > bestv
            best, bestv = sol.u, v
        end
    end
    return best
end
