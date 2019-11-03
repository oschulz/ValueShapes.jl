# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ElasticArrays, ArraysOfArrays


@testset "array_shape" begin
    @test_throws ArgumentError @inferred(ValueShapes._valshapeoftype(Vector{Int}))

    @test @inferred(valshape(rand(3))) == ArrayShape{Float64,1}((3,))
    @test @inferred(valshape(rand(3, 4, 5))) == ArrayShape{Float64,3}((3, 4, 5))

    @inferred(ValueShapes.nonabstract_eltype(ArrayShape{Complex,3}((3, 4, 5)))) == Complex{Float64}

    @test @inferred(totalndof(ArrayShape{Float64,1}((3,)))) == 3
    @test @inferred(totalndof(ArrayShape{Complex,3}((3, 4, 5)))) == 120

    @test valshape(@inferred(Array(undef, ArrayShape{Complex,3}((3, 4, 5))))) == ArrayShape{Complex{Float64},3}((3, 4, 5))
    @test typeof(@inferred(Array(undef, ArrayShape{Complex,3}((3, 4, 5))))) == Array{Complex{Float64},3}
    @test size(@inferred(Array(undef, ArrayShape{Complex,3}((3, 4, 5))))) == (3, 4, 5)
    @test typeof(@inferred(Array{Float32}(undef, ArrayShape{Real,1}((3,))))) == Array{Float32,1}

    @test valshape(@inferred(ElasticArray(undef, ArrayShape{Complex,3}((3, 4, 5))))) == ArrayShape{Complex{Float64},3}((3, 4, 5))
    @test typeof(@inferred(ElasticArray(undef, ArrayShape{Complex,3}((3, 4, 5))))) == ElasticArray{Complex{Float64},3,2}
    @test size(@inferred(ElasticArray(undef, ArrayShape{Complex,3}((3, 4, 5))))) == (3, 4, 5)
    @test typeof(@inferred(ElasticArray{Float32}(undef, ArrayShape{Real,2}((3,4))))) == ElasticArray{Float32,2,1}

    shape = ArrayShape{Real}(2,3)
    A = @inferred(shape(undef))
    @test typeof(A) == Array{Float64,2}
    @test size(A) == (2, 3)
    @test @inferred(valshape(A)) == ArrayShape{Float64}(2,3)
    @test @inferred(shape([1, 2, 3, 4, 5, 6])) == [1 3 5; 2 4 6]
    @test_throws ArgumentError shape([1, 2, 3, 4, 5, 6, 7])

    @test eltype(@inferred(Vector{Float32}(undef, shape))) == Float32
    @test eltype(eltype(@inferred(VectorOfSimilarVectors{Float32}(shape)))) == Float32

    @test valshape(shape.(push!(@inferred(VectorOfSimilarVectors(shape)), @inferred(Vector(undef, shape))))[1]) == valshape(shape(undef))

    let
        A = collect(1:8)
        ac = ValueAccessor(ArrayShape{Real}(3), 2)
        @test @inferred(getindex(A, ac)) == [3, 4, 5]
        @test @inferred(view(A, ac)) == [3, 4, 5]
        @test @inferred(setindex!(A, [7, 2, 6], ac)) === A
        @test A[ac] == [7, 2, 6]
    end

    let
        A = collect(1:6*6)
        ac = ValueAccessor(ArrayShape{Real}(2,3), 17)
        @test @inferred(getindex(A, ac)) == [18 20 22; 19 21 23]
        @test @inferred(view(A, ac)) == [18 20 22; 19 21 23]
        @test @inferred(setindex!(A, [6 4 3; 2 1 5], ac)) === A
        @test A[ac] == [6 4 3; 2 1 5]
    end

    let
        A = collect(reshape(1:6*6, 6, 6))
        ac1 = ValueAccessor(ArrayShape{Real}(2), 2)
        ac2 = ValueAccessor(ArrayShape{Real}(3), 3)
        @test @inferred(getindex(A, ac1, ac2)) == [21 27 33; 22 28 34]
        @test @inferred(view(A, ac1, ac2)) == [21 27 33; 22 28 34]
        @test @inferred(setindex!(A, [6 4 3; 2 1 5], ac1, ac2)) === A
        @test A[ac1, ac2] == [6 4 3; 2 1 5]
    end
end
