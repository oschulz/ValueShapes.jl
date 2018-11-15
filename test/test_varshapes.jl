# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ArraysOfArrays
import StatsBase, TypedTables


@testset "varshapes" begin
    @testset "VarShapes" begin
        get_c(x) = x.c

        parvalues = VectorOfSimilarVectors(reshape(collect(1:22), 11, 2))
        ref_table = TypedTables.Table(a = [[1 3 5; 2 4 6], [12 14 16; 13 15 17]], b = [7, 18], c = [[8, 9, 10, 11], [19, 20, 21, 22]])

        parshapes = @inferred VarShapes(a = (2,3), b = (), c = (4,))
        @test @inferred(get_c(parshapes)) === ValueShapes._accessors(parshapes).c
        @test @inferred(Base.propertynames(parshapes)) == (:a, :b, :c)
        @test @inferred(StatsBase.dof(parshapes)) == 11

        @test @inferred(parshapes(parvalues[1])) == ref_table[1]
        @test @inferred(parshapes(parvalues)) == ref_table
    end

    @testset "examples" begin
        @test begin
            parshapes = VarShapes(a = (2,3), b = (), c = (4,))
            data = VectorOfSimilarVectors{Int}(parshapes)
            resize!(data, 10)
            rand!(flatview(data), 0:99)
            table = parshapes(data)
            fill!(table.b, 42)
            all(x -> x == 42, view(flatview(data), 7, :))
        end
    end
end
