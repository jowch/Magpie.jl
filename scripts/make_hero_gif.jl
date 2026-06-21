# scripts/make_hero_gif.jl — run with: julia --project=docs scripts/make_hero_gif.jl
# ponytail: standalone artifact generator; Plots stays out of the package deps.

ENV["GKSwstype"] = "100"  # GR headless rendering (no display required)

using Plots; gr()
using Magpie, KernelFunctions, LinearAlgebra, Random

Random.seed!(42)

f(x) = norm(x) - 1.0   # signed distance to the unit circle

al = ActiveLearner(
    ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise=1e-4),
    Straddle(h=0.0),
)

box = Box([-2.0, -2.0], [2.0, 2.0])

# Cold start: 10 random observations over the domain
for x in [4 .* rand(2) .- 2 for _ in 1:10]
    observe!(al, x, f(x))
end

# Fit hyperparameters once after cold start
fit!(al)

# Evaluation grid for posterior mean and true contour
xs = range(-2, 2; length=40)
ys = range(-2, 2; length=40)

nframes = 30

anim = @animate for _ in 1:nframes
    # Step one acquisition
    x = acquire(al; over=box)
    observe!(al, x, f(x))

    # Compute posterior mean on the grid
    g = posterior_gp(al)
    Z_mean = [predmean(g, [xi, yj]) for yj in ys, xi in xs]

    # Compute the true signed-distance field for the honest contour overlay
    Z_true = [f([xi, yj]) for yj in ys, xi in xs]

    # Query points accumulated so far
    pts = queried_points(al)
    px  = [p[1] for p in pts]
    py  = [p[2] for p in pts]

    n = length(pts)

    heatmap(xs, ys, Z_mean;
        c            = :RdBu,
        clim         = (-2, 2),
        aspect_ratio = :equal,
        xlims        = (-2, 2),
        ylims        = (-2, 2),
        size         = (560, 420),
        title        = "active learning · n=$(n)",
        xlabel       = "x₁",
        ylabel       = "x₂",
        colorbar     = false,
    )
    contour!(xs, ys, Z_true; levels=[0.0], lw=2, lc=:black)  # the TRUE unit circle, not the posterior's own contour
    scatter!(px, py;
        ms    = 4,
        mc    = :white,
        msw   = 1,
        label = false,
    )
end

outdir = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(outdir)
outpath = joinpath(outdir, "hero.gif")
gif(anim, outpath; fps=8)

println("Wrote: $outpath")
println("Size:  $(filesize(outpath)) bytes  ($(round(filesize(outpath)/1024; digits=1)) KB)")
