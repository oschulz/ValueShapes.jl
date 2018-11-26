# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [fixdoctests]
#
# for local builds.

using Documenter
using ShapesOfVariables

makedocs(
    sitename = "ShapesOfVariables",
    modules = [ShapesOfVariables],
    format = :html,
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://oschulz.github.io/ShapesOfVariables.jl/stable/",
)

deploydocs(
    repo = "github.com/oschulz/ShapesOfVariables.jl.git"
)
