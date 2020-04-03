# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ValueShapes

using ArraysOfArrays
using Distributions
using ElasticArrays
using FillArrays
using Random
using Statistics
using StatsBase

import EmpiricalDistributions
import IntervalSets
import Tables
import TypedTables

include("value_shape.jl")
include("value_accessor.jl")
include("scalar_shape.jl")
include("array_shape.jl")
include("const_value_shape.jl")
include("named_tuple_shape.jl")
include("functions.jl")
include("distributions.jl")
include("const_value_dist.jl")
include("named_tuple_dist.jl")


"""
ValueShapes provides Julia types to describe the shape of values, like
scalars, arrays and structures.
"""
ValueShapes

end # module
