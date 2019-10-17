# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ValueShapes

using ArraysOfArrays
using ElasticArrays
using FillArrays
using Requires

import TypedTables

include("value_shape.jl")
include("value_accessor.jl")
include("scalar_shape.jl")
include("array_shape.jl")
include("const_value_shape.jl")
include("named_tuple_shape.jl")

function __init__()
    @require Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f" include("distributions_support.jl")
end

end # module
