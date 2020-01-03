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



        # Don't hardcode these numbers like length.
        @test @inferred(length(shape) == 5)
        @test @inferred(keys(shape) == (:a, :b, :c, :x, :y))

        # Get properties from ValueShapes.getproperty() function and test attributes
        properties_valueaccessor = getproperty(shape, :_accessors)
        for i in 1:length(keys(named_shapes))
            # src only has getindex(::NamedTupleShape, ::Integer). It also works with Symbol. Good practice to also have symbol function?
            @test @inferred(getindex(named_shapes, i) == named_shapes[i])
        end

        # Test :_flatdof. Current concern: is shape.c.len supposed to be a dof, or the length? Length should be 1. but it return 0. 
        let expected_flatdof = 0, actual_flatdof = getproperty(shape, :_flatdof)
            for va in getproperty(shape, :_accessors)
                expected_flatdof += va.len
            end
            @test expected_flatdof == actual_flatdof
        end
        
        for (k,v) in zip(keys(named_shapes), named_shapes)
       #    @test properties_valueaccessor[k].len == length(named_shapes[k])
            @test length(properties_valueaccessor[k]) == length(named_shapes[k]) # <=== length(va) != va.len
       #    @test properties_valueaccessor[k].len == length(named_shapes[k])
       #    @test properties_valueaccessor[k].offset
       #    scalars don't have shape
       #    @test properties_valueaccessor[k].shape.dims
        end



        @test @inferred(ValueShapes.default_unshaped_eltype(NamedTupleShape(a = ScalarShape{Int}(), b = ArrayShape{Float32}(2, 3)))) == Float32
        @test @inferred(ValueShapes.default_unshaped_eltype(shape)) == Float64

        @test @inferred(ValueShapes.shaped_type(shape)) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test @inferred(ValueShapes.shaped_type(shape, Float32)) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float32,2},Float32,Float64,Array{Int,2},Array{Float32,1}}}

        @test @inferred(get_y(shape)) === ValueShapes._accessors(shape).y
        @test @inferred(Base.propertynames(shape)) == (:a, :b, :c, :x, :y)
        @test @inferred(totalndof(shape)) == 11

        @test @inferred(shape(data[1])[]) == ref_table[1]
        @test @inferred(broadcast(shape, data)) == ref_table

        @test @inferred(merge((foo = 42,), shape)) == merge((foo = 42,), named_shapes)
        @test @inferred(NamedTupleShape(;shape...)) == shape

        @test typeof(@inferred(shape(undef))) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test typeof(@inferred(valshape(shape(undef)))) <: NamedTupleShape
        @test typeof(valshape(shape(undef))(undef)) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test @inferred(shape(collect(1:11))[]) == (a = [1 3 5; 2 4 6], b = 7, c = 4.2, x = [11 21; 12 22], y = [8, 9, 10, 11])
        @test_throws ArgumentError shape(collect(1:12))

        @test valshape(shape.(push!(@inferred(VectorOfSimilarVectors{Float64}(shape)), @inferred(Vector{Float64}(undef, shape))))[1]) == valshape(shape(undef))

        @testset "ValueShapes.ShapedAsNT" begin
            UA = copy(data[1])
            @test @inferred(size(@inferred(ValueShapes.ShapedAsNT(UA, shape)))) == ()
            A = ValueShapes.ShapedAsNT(UA, shape)



            @test @inferred(IndexStyle(A) == IndexLinear())
#           @test axes(A,1).stop == size(A)[1]
#           @test axes(A,2).stop == size(A)[2]


            @test @inferred(propertynames(A)) == (:a, :b, :c, :x, :y)
            @test propertynames(A, true) == (:a, :b, :c, :x, :y, :__internal_data, :__internal_valshape)
            @test @inferred(get_y(A)) == [8, 9, 10, 11]

            @test typeof(A.b) <: AbstractArray{Int,0}

            @test @inferred(valshape(A)) === shape

            @test @inferred(stripscalar(A)) == A[]
            @test @inferred(stripscalar(A.a)) == A.a
            @test @inferred(stripscalar(A.b)) == A.b[]
            @test @inferred(stripscalar(A.c)) == A.c
            @test @inferred(stripscalar(A.x)) === A.x
            @test @inferred(unshaped(A.y)) === A.y

            @test @inferred(unshaped(A)) === UA
            @test @inferred(unshaped(A.a)) == view(UA, 1:6)
            @test @inferred(unshaped(A.b)) == view(UA, 7:7)
            @test @inferred(unshaped(A.y)) == view(UA, 8:11)

            @test size(@inferred similar(A)) == size(A)

            @test @inferred(copy(A)) == A
            @test typeof(copy(A)) == typeof(A)

            @test @inferred((X -> (Y = copy(X); Y.a = [5 3 5; 9 4 5]; unshaped(Y)))(A)) == [5, 9, 3, 4, 5, 5, 7, 8, 9, 10, 11]
            @test @inferred((X -> (Y = copy(X); Y.b = 9; unshaped(Y)))(A)) == [1, 2, 3, 4, 5, 6, 9, 8, 9, 10, 11]
            @test @inferred((X -> (Y = copy(X); Y.c = 4.2; unshaped(Y)))(A)) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            @test_throws ArgumentError (X -> (Y = copy(X); Y.c = 4.3; unshaped(Y)))(A)
            @test @inferred((X -> (Y = copy(X); Y.y = [4, 7, 5, 6]; unshaped(Y)))(A)) == [1, 2, 3, 4, 5, 6, 7, 4, 7, 5, 6]

            x = (a = [5 3 5; 9 4 5], b = 9, c = 4.2, x = [11 21; 12 22], y = [4, 7, 5, 6])
            @test (B = copy(A); B[] = x; B[1]) == x
            @test (B = copy(A); B[1] = x; B[]) == x
            @test_throws ArgumentError copy(A)[] = (a = [5 3 5; 9 4 5], b = 9, c = 4.2, x = [11 21; 12 23], y = [4, 7, 5, 6])
        end

        @testset "ValueShapes.ShapedAsNTArray" begin
            UA = Array(data)
            @test @inferred(size(@inferred(ValueShapes.ShapedAsNTArray(UA, shape)))) == (2,)
            A = ValueShapes.ShapedAsNTArray(UA, shape)

            @inferred typeof(@inferred broadcast(shape, data)) == typeof(A)
            @test shape.(data) == A

            @test @inferred(propertynames(A)) == (:a, :b, :c, :x, :y)
            @test propertynames(A, true) == (:a, :b, :c, :x, :y, :__internal_data, :__internal_elshape)
            @test @inferred(get_y(A)) == [[8, 9, 10, 11], [19, 20, 21, 22]]

            @test @inferred(elshape(A)) === shape

            @test @inferred(broadcast(unshaped, A)) === UA

            @test @inferred(A[1]) == (a = [1 3 5; 2 4 6], b = 7, c = 4.2, x = [11 21; 12 22], y = [8, 9, 10, 11])
            @test @inferred(view(A, 2)[]) == A[2]

            @test @inferred(append!(copy(A), copy(A)))[3:4] == @inferred(A[1:2])
            @test @inferred(vcat(A, A))[3:4] == @inferred(A[1:2])

            @test size(@inferred similar(A)) == size(A)

            @test @inferred(copy(A)) == A
            @test typeof(copy(A)) == typeof(A)

            @test @inferred(TypedTables.Table(A)) == A
            @test typeof(@inferred flatview(TypedTables.Table(shape.(data)).y)) == Array{Int,2}

            A_zero() = shape.(nestedview(zeros(totalndof(shape), 2)))
            @test (B = A_zero(); B[:] = A; B) == A
            @test (B = A_zero(); B[:] = TypedTables.Table(A); B) == A
        end
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
