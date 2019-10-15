# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ValueShapes" begin
    include("test_value_shape.jl")
    include("test_value_accessor.jl")
    include("test_scalar_shape.jl")
    include("test_array_shape.jl")
    include("test_const_value_shape.jl")
    include("test_named_tuple_shape.jl")
    include("test_distributions_support.jl")
end # testset
