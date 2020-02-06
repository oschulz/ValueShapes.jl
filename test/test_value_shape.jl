# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ElasticArrays
using ArraysOfArrays
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

        # @test Vector{Real}(undef, ArrayShape{Real}((2,1))) ==  # weird typing going on with default_unshaped_eltype

        arrshape = ArrayShape{Real, 2}((2,3))
        vec = Vector{Real}(undef, arrshape)
        @test @inferred(length(vec)) == 6
        @test @inferred(size(vec)) == (6,)
        
        data1 = [1;2;3;4;7;8;9]
        scalarshape = ScalarShape{Real}()
        ntshape = NamedTupleShape(a=arrshape, b=scalarshape)
        shapedasnt = ntshape(data1)       
        @test stripscalar(Ref(shapedasnt)) == Ref(shapedasnt)[]

        @test_throws ArgumentError Broadcast.broadcastable(ntshape) 

        named_shapes = (
            a = ArrayShape{Real}(2, 3),
            b = ScalarShape{Real}(),
            c = ConstValueShape(4.2),
            x = ConstValueShape([11 21; 12 22]),
            y = ArrayShape{Real}(4)
        )
        shape = NamedTupleShape(;named_shapes...)
        data2 = VectorOfSimilarVectors(reshape(collect(1:22), 11, 2))
        @test_throws ArgumentError ValueShapes._checkcompat_inner(ntshape, data2)
        @test ValueShapes._checkcompat_inner(shape, data2) == nothing
    end
end
