# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ValueShapes

using ArraysOfArrays
using ElasticArrays
using MacroTools

import StatsBase
import TypedTables

include("varshapes.jl")

end # module
