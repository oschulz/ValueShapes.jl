# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ValueShapes" begin
    include("test_valueshape.jl")
    include("test_varshapes.jl")
end # testset
