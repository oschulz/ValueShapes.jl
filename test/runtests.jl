# This file is a part of ShapesOfVariables.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ShapesOfVariables" begin
    include("test_valueshape.jl")
    include("test_varshapes.jl")
end # testset
