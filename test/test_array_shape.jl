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
    @test @inferred(length(shape)) == shape.dims[1]*shape.dims[2]
    @test @inferred(length(shape)) == prod(shape.dims)
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

    let A = collect(reshape(1:5*5*5, 5, 5, 5))
        ac1 = ValueAccessor(ArrayShape{Real}(size(A)[1]), 0)
        ac2 = ValueAccessor(ArrayShape{Real}(1), 0)
        ac3 = ValueAccessor(ArrayShape{Real}(1), 1)
        ac4 = ValueAccessor(ArrayShape{Real}(1), 2)

        @test @inferred(getindex(A, ac1, ac1, ac1)) == A
        @test @inferred(getindex(A, ac2, ac2, ac2))[1] == A[1]
        @test @inferred(getindex(A, ac4, ac4, ac4))[1] == 63

        @test @inferred(view(A, ac1, ac1, ac1)) == A
        @test @inferred(view(A, ac2, ac2, ac2))[1] == A[1]
        @test @inferred(view(A, ac4, ac4, ac4))[1] == 63

        first_layer = @inferred(getindex(A, ac1, ac1, ac2))
        setindex!(A, first_layer, ac1, ac1, ac3)
        @test @inferred(A[:,:,1]) == @inferred(A[:,:,2])
    end

    let d1 = [11, 12, 13, 14], d2 = [21, 22]
        d = vcat(d1, d2)
        reshaped = shape(d)
        @test reshaped == reshape(d, (2,3))
    end
end

@testset "broadcasting and copy" begin
    data1d = [rand(4), rand(4), rand(4), rand(4)]
    data2d = [[rand(4,)] [rand(4,)] [rand(4,)]]

    VoV = VectorOfVectors(data1d)

    shape1 = ArrayShape{Float64, 1}((4,))
    shape2 = ArrayShape{Float64}(1,4)
    shape3 = ArrayShape{Float64}(2,2)

    shape1_VoV_bcast = broadcast(shape1, VoV)
    shape2_VoV_bcast = broadcast(shape2, VoV)
    shape3_VoV_bcast = broadcast(shape3, VoV)

    shape1_bcast = broadcast(shape1, data1d)
    shape1_data_bcast = broadcast(shape1, data1d)
    shape1_data_dcast = shape1.(data1d)
    shape2_data_bcast = broadcast(shape2, data1d)
    shape3_data_bcast = broadcast(shape3, data1d)

    @test shape1_data_dcast == shape1_data_bcast
    @test isapprox(shape1_VoV_bcast, shape1_data_bcast)
    @test isapprox(shape2_VoV_bcast, shape2_data_bcast)
    @test isapprox(shape3_VoV_bcast, shape3_data_bcast)

    AoSV = ArrayOfSimilarVectors{Float64}(data1d)
    AoSA = ArrayOfSimilarArrays{Float64}(data2d)

    shaped_AoSV_bcast = broadcast(shape1, AoSV)
    shaped_AoSV_dcast = shape1.(AoSV)

    shaped_AoSA_bcast = broadcast(shape1, AoSA)
    shaped_AoSA_dcast = shape1.(AoSA)

    @test shaped_AoSV_bcast == AoSV
    @test shaped_AoSV_dcast == shaped_AoSV_bcast
    @test shape1.(VoV) == VoV

    shape1_bcast = broadcast(shape1, data2d)
    @test shape1_bcast == shape1.(data2d)
    for i in 1:length(data2d)
        @test shape1_bcast[i] == data2d[i]
    end
    unshaped1 = unshaped.(shaped_AoSA_bcast)
    @test unshaped1 == data2d
end
