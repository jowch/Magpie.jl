mutable struct ActiveLearner{A}
    gp::AbstractGPModel; acq::A
    Xs::Vector{Any}; Ys::Vector{Any}; acq_vals::Vector{Float64}
end
ActiveLearner(gp, acq) = ActiveLearner{typeof(acq)}(gp, acq, Any[], Any[], Float64[])

posterior_gp(al::ActiveLearner) = al.gp
queried_points(al::ActiveLearner) = copy(al.Xs)
all_data(al::ActiveLearner) = (copy(al.Xs), copy(al.Ys))

_isbatch(X) = X isa AbstractVector && !isempty(X) && first(X) isa AbstractVector
function observe!(al::ActiveLearner, X, Y)
    Xv = _isbatch(X) ? X : [X]
    Yv = Y isa AbstractVector ? Y : [Y]
    append!(al.Xs, Xv); append!(al.Ys, Yv)
    al.gp = Magpie.update(al.gp, Xv, Yv)
    return al
end
fit!(al::ActiveLearner; restarts::Int=1) = (al.gp = Magpie.fit(al.gp; restarts=restarts); al)

function acquire(al::ActiveLearner; over, maximizer=default_for(over), q::Int=1)
    q == 1 || error("batch acquisition (q>1) not yet implemented; use q=1")
    return acquire(al.gp, resample(al.acq); over=over, maximizer=maximizer)
end

function run!(al::ActiveLearner, oracle; budget::Int, over, stop = al -> false, refit_every::Int=0)
    for t in 1:budget
        stop(al) && break
        a = resample(al.acq)
        x = acquire(al.gp, a; over=over)
        push!(al.acq_vals, a(al.gp, x))
        observe!(al, x, oracle(x))
        refit_every > 0 && t % refit_every == 0 && fit!(al)
    end
    return al
end
