# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions

#@testset "const_value_dist" begin
@testset "test" begin
    cvd = ConstValueDist(42)
    shape = varshape(cvd)
    @test @inferred totalndof(shape) == 0
end
