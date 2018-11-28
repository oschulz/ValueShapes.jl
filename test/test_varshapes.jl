# This file is a part of ShapesOfVariables.jl, licensed under the MIT License (MIT).

using ShapesOfVariables
using Test

using Random
using ArraysOfArrays
import TypedTables


@testset "varshapes" begin
    @testset "VarShapes" begin
        get_y(x) = x.y

        data = VectorOfSimilarVectors(reshape(collect(1:22), 11, 2))
        ref_table = TypedTables.Table(
            a = [[1 3 5; 2 4 6], [12 14 16; 13 15 17]],
            b = [7, 18], c = [4.2, 4.2],
            x = Matrix[[11 21; 12 22], [11 21; 12 22]],
            y = [[8, 9, 10, 11], [19, 20, 21, 22]]
        )

        varshapes = @inferred VarShapes(a = (2,3), b = (), c = 4.2, x = [11 21; 12 22], y = (4,))
        @test @inferred(get_y(varshapes)) === ShapesOfVariables._accessors(varshapes).y
        @test @inferred(Base.propertynames(varshapes)) == (:a, :b, :c, :x, :y)
        @test @inferred(flatdof(varshapes)) == 11

        @test @inferred(varshapes(data[1])) == ref_table[1]
        @test @inferred(varshapes(data)) == ref_table
    end

    @testset "examples" begin
        @test begin
            varshapes = VarShapes(a = (2,3), b = (), c = 4.2, x = [11 21; 12 22], y = (4,))
            data = VectorOfSimilarVectors{Int}(varshapes)
            resize!(data, 10)
            rand!(flatview(data), 0:99)
            table = varshapes(data)
            fill!(table.b, 42)
            all(x -> x == 42, view(flatview(data), 7, :))
        end
    end
end
