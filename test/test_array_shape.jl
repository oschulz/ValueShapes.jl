# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ElasticArrays, ArraysOfArrays


@testset "array_shape" begin
    @test_throws ArgumentError @inferred(ValueShapes._valshapeoftype(Vector{Int}))

    @test @inferred(valshape(rand(3))) == ArrayShape{Float64,1}((3,))
    @test @inferred(valshape(rand(3, 4, 5))) == ArrayShape{Float64,3}((3, 4, 5))

    @test @inferred(ValueShapes.default_unshaped_eltype(ArrayShape{Complex,3}((3, 4, 5)))) == Float64
    @test @inferred(ValueShapes.default_unshaped_eltype(ArrayShape{Complex{Float32},3}((3, 4, 5)))) == Float32

    @test @inferred(ValueShapes.shaped_type(ArrayShape{Real}(2, 3, 4))) == Array{Float64, 3}
    @test @inferred(ValueShapes.shaped_type(ArrayShape{Complex}(2))) == Array{Complex{Float64}, 1}
    @test @inferred(ValueShapes.shaped_type(ArrayShape{Complex{Real}}(2), Float32)) == Array{Complex{Float32}, 1}
    @test @inferred(ValueShapes.shaped_type(ArrayShape{Complex{Int16}}(2, 3))) == Array{Complex{Int16}, 2}

    @test @inferred(totalndof(ArrayShape{Float64,1}((3,)))) == 3
    @test @inferred(totalndof(ArrayShape{Complex,3}((3, 4, 5)))) == 120

    @test size(@inferred(Vector{Float64}(undef, ArrayShape{Complex}((2, 1, 3))))) == (2 * 2*1*3,)
    @test size(flatview(@inferred(VectorOfSimilarVectors{Float32}(ArrayShape{Complex}((2, 1, 3)))))) == (2 * 2*1*3, 0)

    shape = ArrayShape{Real}(2,3)
    A = @inferred(shape(undef))
    @test typeof(A) == Array{Float64,2}
    @test size(A) == (2, 3)
    @test @inferred(valshape(A)) == ArrayShape{Float64}(2,3)
    @test @inferred(shape([1, 2, 3, 4, 5, 6])) == [1 3 5; 2 4 6]
    @test_throws ArgumentError shape([1, 2, 3, 4, 5, 6, 7])

    @test eltype(@inferred(Vector{Float32}(undef, shape))) == Float32
    @test eltype(eltype(@inferred(VectorOfSimilarVectors{Float32}(shape)))) == Float32

    @test valshape(shape.(push!(@inferred(VectorOfSimilarVectors{Float64}(shape)), @inferred(Vector{Float64}(undef, shape))))[1]) == valshape(shape(undef))

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
