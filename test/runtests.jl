# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ValueShapes" begin
    include("test_aqua.jl")
    include("test_value_shape.jl")
    include("test_value_accessor.jl")
    include("test_scalar_shape.jl")
    include("test_array_shape.jl")
    include("test_const_value_shape.jl")
    include("test_named_tuple_shape.jl")
    include("test_functions.jl")
    include("test_distributions.jl")
    include("test_const_value_dist.jl")
    include("test_named_tuple_dist.jl")
    include("test_reshaped_dist.jl")
    # Mooncake's compilation latency on Julia v1.10 makes its tests impractically
    # slow, run them on newer Julia versions only:
    VERSION >= v"1.11" && include("test_mooncake.jl")
end # testset
