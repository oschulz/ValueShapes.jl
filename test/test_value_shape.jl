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

        @test @inferred(elshape(Complex(1.0, 2.0))) == ScalarShape{Complex{Float64}}()
        @test @inferred(elshape([[3, 5], [3, 2]])) == ArrayShape{Int,1}((2,))

        @test ValueShapes.stripscalar(Ref(ScalarShape{Real})) == ScalarShape{Real}

#       @test Vector{Real}(undef, ArrayShape{Real}((2,1))) ==  # weird typing going on with default_unshaped_eltype

    end
end
