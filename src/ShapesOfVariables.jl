# This file is a part of ShapesOfVariables.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ShapesOfVariables

using ArraysOfArrays
using ElasticArrays
using FillArrays
using MacroTools

import TypedTables

include("valueshape.jl")
include("varshapes.jl")

end # module
