# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ArraysOfArrays
import TypedTables


@testset "const_value_shape" begin
    @inferred(size(ConstValueShape(42))) == ()
    @inferred(eltype(ConstValueShape(42))) == Int
    @inferred(totalndof(ConstValueShape(42))) == 0

    @inferred(size(ConstValueShape(rand(2,3)))) == (2,3)
    @inferred(eltype(ConstValueShape(rand(Float32,2,3)))) == Float32
    @inferred(totalndof(ConstValueShape(rand(2,3)))) == 0

    @test @inferred(ConstValueShape([1 4; 3 2])(undef)) == [1 4; 3 2]
    @test @inferred(ConstValueShape([1 4; 3 2])(Int[])) == [1 4; 3 2]

    shape = ConstValueShape([1 4; 3 2])
    @test typeof(@inferred(Vector(undef, shape))) == Vector{Int}
    @test size(@inferred(Vector(undef, shape))) == (0,)
end
