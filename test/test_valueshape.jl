# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ElasticArrays
import TypedTables


@testset "valueshape" begin
    @testset "ValueShapes.default_datatype" begin
        @test @inferred(ValueShapes.default_datatype(Integer)) == Int
        @test @inferred(ValueShapes.default_datatype(AbstractFloat)) == Float64
        @test @inferred(ValueShapes.default_datatype(Real)) == Float64
        @test @inferred(ValueShapes.default_datatype(Complex)) == Complex{Float64}
    end

    @testset "ScalarShape" begin
        @test @inferred(shapeoftype(Int)) == ScalarShape{Int}()
        @test @inferred(shapeoftype(Complex{Float64})) == ScalarShape{Complex{Float64}}()


        @test @inferred(shapeof(3)) == ScalarShape{Int}()
        @test @inferred(shapeof("foo")) == ScalarShape{String}()

        @test @inferred(totalndof(ScalarShape{Int}())) == 1
        @test @inferred(totalndof(ScalarShape{Complex{Float64}}())) == 2
    end

    @testset "ArrayShape" begin
        @test_throws ArgumentError @inferred(shapeoftype(Vector{Int}))

        @test @inferred(shapeof(rand(3))) == ArrayShape{Float64,1}((3,))
        @test @inferred(shapeof(rand(3, 4, 5))) == ArrayShape{Float64,3}((3, 4, 5))

        @inferred(ValueShapes.nonabstract_eltype(ArrayShape{Complex,3}((3, 4, 5)))) == Complex{Float64}

        @test @inferred(totalndof(ArrayShape{Float64,1}((3,)))) == 3
        @test @inferred(totalndof(ArrayShape{Complex,3}((3, 4, 5)))) == 120

        @test shapeof(@inferred(Array(undef, ArrayShape{Complex,3}((3, 4, 5))))) == ArrayShape{Complex{Float64},3}((3, 4, 5))
        @test typeof(@inferred(Array(undef, ArrayShape{Complex,3}((3, 4, 5))))) == Array{Complex{Float64},3}
        @test size(@inferred(Array(undef, ArrayShape{Complex,3}((3, 4, 5))))) == (3, 4, 5)
        @test typeof(@inferred(Array{Float32}(undef, ArrayShape{Real,1}((3,))))) == Array{Float32,1}

        @test shapeof(@inferred(ElasticArray(undef, ArrayShape{Complex,3}((3, 4, 5))))) == ArrayShape{Complex{Float64},3}((3, 4, 5))
        @test typeof(@inferred(ElasticArray(undef, ArrayShape{Complex,3}((3, 4, 5))))) == ElasticArray{Complex{Float64},3,2}
        @test size(@inferred(ElasticArray(undef, ArrayShape{Complex,3}((3, 4, 5))))) == (3, 4, 5)
        @test typeof(@inferred(ElasticArray{Float32}(undef, ArrayShape{Real,2}((3,4))))) == ElasticArray{Float32,2,1}
    end

    @testset "ConstValueShape" begin
        @inferred(size(ConstValueShape(42))) == ()
        @inferred(eltype(ConstValueShape(42))) == Int
        @inferred(totalndof(ConstValueShape(42))) == 0

        @inferred(size(ConstValueShape(rand(2,3)))) == (2,3)
        @inferred(eltype(ConstValueShape(rand(Float32,2,3)))) == Float32
        @inferred(totalndof(ConstValueShape(rand(2,3)))) == 0
    end
end
