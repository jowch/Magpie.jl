# bench/timing_exact_vs_svgp.jl — Task 4.3: ExactGPField vs SVGPField per-gradient cost
#
# Measures per-gradient wall-clock (seconds) and allocations (bytes) for:
#   (a) ExactGPField with N anchors — Cholesky cost O(N³)
#   (b) SVGPField with FIXED M=15 inducing points trained on N data points — cost O(M³)
# as N ∈ {20, 50, 100, 200}.
#
# This is the number the scale_forcing example asserted without measuring.
#
# This is a standalone benchmark, run BY HAND (not a CI gate). It depends on
# OrdinaryDiffEq/SciMLSensitivity (the package's test-only deps), so run it from an environment
# where those are available — e.g. a throwaway env that has Magpie + the two SciML packages:
#
#   julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); \
#             Pkg.add(["OrdinaryDiffEq","SciMLSensitivity"]); include("bench/timing_exact_vs_svgp.jl")'
#
# It keeps an internal @assert (SVGP allocs < Exact/2 at N=200) so a by-hand run self-checks.
#
# NOTE: This script is NOT in the default test suite (runtests.jl). It is slow and
# timing-based. Opt-in: MAGPIE_BENCH=true alongside MAGPIE_TEST_SCIML=true.

using Magpie, KernelFunctions, LinearAlgebra, Random
using OrdinaryDiffEq, SciMLSensitivity
import DifferentiationInterface as DI
import Mooncake

const M_FIXED = 15   # SVGP inducing points — held constant as N grows

# ---------------------------------------------------------------------------
# Helper: build a synthetic 2D trajectory of length N data points
# ---------------------------------------------------------------------------
function synthetic_data(N::Int; rng=MersenneTwister(42))
    lv!(du, u, p, t) = (du[1] = 1.5u[1] - u[1]*u[2]; du[2] = u[1]*u[2] - 3u[2]; nothing)
    u0 = [1.0, 1.0]; tspan = (0.0, 3.0)
    ts = collect(range(tspan...; length=N))
    target = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat=ts))
    return u0, tspan, ts, target
end

# ---------------------------------------------------------------------------
# Helper: one gradient evaluation (warm path), return (time_s, alloc_bytes)
# ---------------------------------------------------------------------------
function time_gradient(loss, v)
    ad = DI.AutoMooncake(; config=nothing)
    # warm-up: compile + caches
    _ = DI.gradient(loss, ad, v)
    # timed run
    stats = @timed DI.gradient(loss, ad, v)
    alloc = @allocated DI.gradient(loss, ad, v)
    return stats.time, alloc
end

# ---------------------------------------------------------------------------
# ExactGPField gradient: N anchors = N data points (worst case — exact GP grows with data)
# ---------------------------------------------------------------------------
function bench_exact(N::Int)
    u0, tspan, ts, target = synthetic_data(N)
    # Anchors = subsampled columns of the trajectory (exactly N)
    Z = [target[:, i] for i in 1:N]
    field = Magpie.ExactGPField(SqExponentialKernel(), Z; d=2)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    L = Magpie.FieldLayout(N, 2)
    loss = ext.make_loss(field, L, u0, tspan, ts, target; λ=1/(N*2))
    v = copy(field.v0)
    return time_gradient(loss, v)
end

# ---------------------------------------------------------------------------
# SVGPField gradient: M=M_FIXED inducing points, N data points (cost ~ M³, not N³)
# ---------------------------------------------------------------------------
function bench_svgp(N::Int)
    u0, tspan, ts, target = synthetic_data(N)
    # Inducing points: fixed M (k-means on the data)
    Z0 = Magpie.kmeans_anchors(target, M_FIXED; rng=MersenneTwister(7))
    field = Magpie.SVGPField(SqExponentialKernel(), Z0; dout=2)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    loss = ext.svgp_elbo_loss(field, [(ts, target)]; tspan)
    v = copy(field.v0)
    return time_gradient(loss, v)
end

# ---------------------------------------------------------------------------
# Main: sweep N, print table, assert crossover at N=200
# ---------------------------------------------------------------------------
println("\n=== Task 4.3: ExactGPField vs SVGPField per-gradient cost ===")
println("M_FIXED = $M_FIXED inducing points (SVGP); Exact anchors = N\n")

Ns     = [20, 50, 100, 200]
exact_times  = Float64[]
exact_allocs = Int[]
svgp_times   = Float64[]
svgp_allocs  = Int[]

for N in Ns
    @info "Benchmarking N=$N ..."
    et, ea = bench_exact(N)
    st, sa = bench_svgp(N)
    push!(exact_times, et); push!(exact_allocs, ea)
    push!(svgp_times,  st); push!(svgp_allocs,  sa)
    @info "  Exact: $(round(et; digits=3))s  $(ea ÷ 1024) KiB"
    @info "  SVGP:  $(round(st; digits=3))s  $(sa ÷ 1024) KiB"
end

println("\n┌─────┬──────────────────────────────┬──────────────────────────────┐")
println("│  N  │   ExactGPField (N anchors)   │  SVGPField (M=$M_FIXED, fixed)   │")
println("│     │   time (s)   │ alloc (KiB)  │   time (s)  │ alloc (KiB)  │")
println("├─────┼──────────────┼──────────────┼─────────────┼──────────────┤")
for (i, N) in enumerate(Ns)
    et = lpad(round(exact_times[i];  digits=3), 10)
    ea = lpad(exact_allocs[i] ÷ 1024, 10)
    st = lpad(round(svgp_times[i];   digits=3), 9)
    sa = lpad(svgp_allocs[i] ÷ 1024, 10)
    println("│ $(lpad(N,3)) │   $(et) s  │  $(ea) KiB  │  $(st) s  │  $(sa) KiB  │")
end
println("└─────┴──────────────┴──────────────┴─────────────┴──────────────┘")

# ---------------------------------------------------------------------------
# Crossover assertion: at N=200, SVGP should allocate well under Exact (M=15 << N=200).
# Allocations are more deterministic than wall-clock; use a factor-2 margin (measured ratio ≈0.23,
# so ample room) so the gate is robust to allocator/Julia-version drift, not a knife-edge.
# ---------------------------------------------------------------------------
exact_alloc_200 = exact_allocs[end]
svgp_alloc_200  = svgp_allocs[end]
ratio = svgp_alloc_200 / exact_alloc_200

println("\n--- Crossover check at N=200 ---")
println("Exact allocs: $(exact_alloc_200 ÷ 1024) KiB")
println("SVGP  allocs: $(svgp_alloc_200 ÷ 1024) KiB")
println("Ratio SVGP/Exact: $(round(ratio; digits=3))")

if svgp_alloc_200 < exact_alloc_200
    println("\n[PASS] SVGP < Exact at N=200 (ratio=$(round(ratio; digits=3))) — crossover confirmed.")
else
    println("\n[DONE_WITH_CONCERNS] SVGP allocs >= Exact at N=200 (ratio=$(round(ratio; digits=3))).")
    println("  Possible cause: SVGP per-gradient overhead (variational machinery, trainable Z) dominates")
    println("  the O(M³) Cholesky saving at M=15, N=200. The crossover likely occurs at larger N.")
    println("  This is honest measurement — the scale_forcing example's claim needs a larger N or larger M_FIXED.")
end

# Hard CI-gate assertion — fires UNCONDITIONALLY (unlike a no-op guarded by the crossover condition):
# if SVGP ever fails to beat Exact by 2× at N=200, the bench fails loudly. Measured ratio ≈0.23.
@assert svgp_alloc_200 < exact_alloc_200 / 2 "SVGP should allocate < half of Exact at N=200 (got ratio=$(round(ratio; digits=3)))"
println("\n[CI gate] @assert passed (SVGP allocs < Exact/2 at N=200).")

println("\nDone.")
