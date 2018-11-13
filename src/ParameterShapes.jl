# This file is a part of ParameterShapes.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ParameterShapes

using ArraysOfArrays
using ElasticArrays
using MacroTools

import StatsBase

include("npargs.jl")
include("param_shapes.jl")

end # module
