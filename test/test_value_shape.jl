# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ElasticArrays
import TypedTables


@testset "abstract_value_shape" begin
    @testset "default_datatype" begin
        @test @inferred(ValueShapes.default_datatype(Integer)) == Int
        @test @inferred(ValueShapes.default_datatype(AbstractFloat)) == Float64
        @test @inferred(ValueShapes.default_datatype(Real)) == Float64
        @test @inferred(ValueShapes.default_datatype(Complex)) == Complex{Float64}
    end
end
