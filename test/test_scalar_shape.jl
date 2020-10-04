# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ArraysOfArrays
import TypedTables


@testset "scalar_shape" begin
    @test @inferred(ValueShapes._valshapeoftype(Int)) == ScalarShape{Int}()
    @test @inferred(ValueShapes._valshapeoftype(Complex{Float64})) == ScalarShape{Complex{Float64}}()

    @test @inferred(valshape(3)) == ScalarShape{Int}()

    @test @inferred(size(ScalarShape{Real}())) == ()
    @test @inferred(size(ScalarShape{Complex}())) == ()

    @test @inferred(size(ScalarShape{Real}())) == ()
    @test @inferred(size(ScalarShape{Complex}())) == ()

    @test @inferred(ValueShapes.default_unshaped_eltype(ScalarShape{Real}())) == Float64
    @test @inferred(ValueShapes.default_unshaped_eltype(ScalarShape{Complex{Real}}())) == Float64
    @test @inferred(ValueShapes.default_unshaped_eltype(ScalarShape{Complex{Int32}}())) == Int32

    @test @inferred(ValueShapes.shaped_type(ScalarShape{Real}())) == Float64
    @test @inferred(ValueShapes.shaped_type(ScalarShape{Real}(), Float32)) == Float32
    @test @inferred(ValueShapes.shaped_type(ScalarShape{Complex}())) == Complex{Float64}
    @test @inferred(ValueShapes.shaped_type(ScalarShape{Complex{Real}}(), Float32)) == Complex{Float32}
    @test @inferred(ValueShapes.shaped_type(ScalarShape{Complex{Int16}}())) == Complex{Int16}

    @test @inferred(totalndof(ScalarShape{Int}())) == 1
    @test @inferred(totalndof(ScalarShape{Complex{Float64}}())) == 2

    @test @inferred(ScalarShape{Real}()(undef)) === zero(Float64)
    @test @inferred(ScalarShape{Complex}()(undef)) === zero(Complex{Float64})

    @test typeof(@inferred(ScalarShape{Real}()([42]))) <: SubArray{Int,0}
    @test @inferred(ScalarShape{Real}()([42])[]) == 42

    @test @inferred (ScalarShape{Int}() <= ScalarShape{Real}()) == true
    @test @inferred (ScalarShape{Real}() <= ScalarShape{Int}()) == false
    @test @inferred (ScalarShape{Real}() >= ScalarShape{Int}()) == true

    let shape = ScalarShape{Real}(), data = [4.2]
        @test @inferred(unshaped(shape(data), shape)) == data
        @test @inferred(unshaped(shape(data)[], shape)) == data
        @test_throws ArgumentError unshaped(shape([3, 4]), shape)
        @test_throws ArgumentError unshaped(shape(data), ScalarShape{Integer}())
    end

    @test let
        shape = ScalarShape{Real}()
        valshape(shape.(push!(@inferred(VectorOfSimilarVectors{Float64}(shape)), @inferred(Vector{Float64}(undef, shape))))[1]) == valshape(shape(undef))
    end

    let
        A = collect(1:6)
        ac = ValueAccessor(ScalarShape{Real}(), 2)
        @test @inferred(getindex(A, ac)) == 3
        @test @inferred(view(A, ac))[] == 3
        @test view(A, ac) isa AbstractArray{<:Real,0}
        @test @inferred(setindex!(A, 7, ac)) === A
        @test A[ac] == 7
    end

    let
        A = collect(reshape(1:6*6, 6, 6))
        ac1 = ValueAccessor(ScalarShape{Real}(), 2)
        ac2 = ValueAccessor(ScalarShape{Real}(), 4)
        @test @inferred(getindex(A, ac1, ac2)) == 27
        @test @inferred(view(A, ac1, ac2))[] == 27
        @test view(A, ac1, ac2) isa AbstractArray{<:Real,0}
        @test @inferred(setindex!(A, 42, ac1, ac2)) === A
        @test A[ac1, ac2] == 42
    end
end
