# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ArraysOfArrays
import TypedTables


@testset "named_tuple_shape" begin
    @testset "functionality" begin
        get_y(x) = x.y

        data = VectorOfSimilarVectors(reshape(collect(1:22), 11, 2))
        ref_table = TypedTables.Table(
            a = [[1 3 5; 2 4 6], [12 14 16; 13 15 17]],
            b = [7, 18],
            c = [4.2, 4.2],
            x = Matrix[[11 21; 12 22], [11 21; 12 22]],
            y = [[8, 9, 10, 11], [19, 20, 21, 22]]
        )

        named_shapes = (
            a = ArrayShape{Real}(2, 3),
            b = ScalarShape{Real}(),
            c = ConstValueShape(4.2),
            x = ConstValueShape([11 21; 12 22]),
            y = ArrayShape{Real}(4)
        )

        shape = @inferred NamedTupleShape(;named_shapes...)
        @test @inferred(NamedTupleShape(named_shapes)) == shape

        @test @inferred(get_y(shape)) === ValueShapes._accessors(shape).y
        @test @inferred(Base.propertynames(shape)) == (:a, :b, :c, :x, :y)
        @test @inferred(totalndof(shape)) == 11

        @test @inferred(shape(data[1])) == ref_table[1]
        @test @inferred(broadcast(shape, data)) == ref_table

        @test @inferred(merge((foo = 42,), shape)) == merge((foo = 42,), named_shapes)
        @test @inferred(NamedTupleShape(;shape...)) == shape

        @test typeof(@inferred(shape(undef))) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test typeof(@inferred(valshape(shape(undef)))) <: NamedTupleShape
        @test typeof(valshape(shape(undef))(undef)) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test @inferred(shape(collect(1:11))) == (a = [1 3 5; 2 4 6], b = 7, c = 4.2, x = [11 21; 12 22], y = [8, 9, 10, 11])
        @test_throws ArgumentError shape(collect(1:12))

        @test valshape(shape.(push!(@inferred(VectorOfSimilarVectors(shape)), @inferred(Vector(undef, shape))))[1]) == valshape(shape(undef))
    end


    @testset "examples" begin
        @test begin
            shape = NamedTupleShape(
                a = ArrayShape{Real}(2, 3),
                b = ScalarShape{Real}(),
                c = ConstValueShape(4.2),
                x = ConstValueShape([11 21; 12 22]),
                y = ArrayShape{Real}(4)
            )
            data = VectorOfSimilarVectors{Int}(shape)
            resize!(data, 10)
            rand!(flatview(data), 0:99)
            table = shape.(data)
            fill!(table.b, 42)
            all(x -> x == 42, view(flatview(data), 7, :))
        end
    end
end
