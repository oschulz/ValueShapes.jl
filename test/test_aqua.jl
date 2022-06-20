# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import ValueShapes

Test.@testset "Aqua tests" begin
    Aqua.test_all(ValueShapes)
end # testset
