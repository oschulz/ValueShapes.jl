# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using LinearAlgebra
using Statistics, StatsBase, Distributions


@testset "reshaped_dist" begin
    scshape = ScalarShape{Real}()

    a1shape = ArrayShape{Real}(3)
    a2shape = ArrayShape{Real}(2, 3)

    ntshape = NamedTupleShape(
        a = ArrayShape{Real}(2, 3),
        b = ScalarShape{Real}(),
        c = ConstValueShape(4.2),
        x = ConstValueShape([11 21; 12 22]),
        y = ArrayShape{Real}(4)
    )

    names = (:a, :b, :c, :x, :y)
    
    _mvnormalrd(shape::AbstractValueShape) = ReshapedDist(MvNormal(fill(2.0, totalndof(shape)), Diagonal(fill(3.0, totalndof(shape)))), shape)
    _mvnormalrd2(shape::AbstractValueShape) = shape(MvNormal(fill(2.0, totalndof(shape)), Diagonal(fill(3.0, totalndof(shape)))))

    @testset "ctors" begin
        @test @inferred(_mvnormalrd(scshape)) isa Distribution{Univariate, Continuous}
        @test @inferred(_mvnormalrd(a1shape)) isa Distribution{Multivariate, Continuous}
        @test @inferred(_mvnormalrd(a2shape)) isa Distribution{Matrixvariate, Continuous}
        @test @inferred(_mvnormalrd(ntshape)) isa Distribution{ValueShapes.NamedTupleVariate{names}, Continuous}

        @test_throws ArgumentError ReshapedDist(MvNormal(Diagonal(fill(3.0, 2))), a2shape)

        @test @inferred(_mvnormalrd2(scshape)) isa ReshapedDist{Univariate, Continuous}
        @test @inferred(_mvnormalrd2(a1shape)) isa MvNormal
        @test @inferred(_mvnormalrd2(a2shape)) isa ReshapedDist{Matrixvariate, Continuous}
        @test @inferred(_mvnormalrd2(ntshape)) isa ReshapedDist{ValueShapes.NamedTupleVariate{names}, Continuous}
    end
    
    @testset "rand" begin
        @test @inferred(rand(_mvnormalrd(scshape))) isa Real
        @test @inferred(rand(_mvnormalrd(scshape), ())) isa AbstractArray{<:Real,0}
        @test @inferred(rand(_mvnormalrd(scshape), 7)) isa AbstractArray{<:Real,1}
        @test @inferred(rand(_mvnormalrd(scshape), (7,))) isa AbstractArray{<:Real,1}
        @test size(rand(_mvnormalrd(scshape), 7)) == (7,)

        @test @inferred(rand(_mvnormalrd(a1shape))) isa AbstractVector{<:Real}
        @test @inferred(rand(_mvnormalrd(a1shape), ())) isa AbstractVector{<:Real}
        @test @inferred(rand(_mvnormalrd(a1shape), 7)) isa AbstractArray{<:AbstractVector{<:Real},1}
        @test @inferred(rand(_mvnormalrd(a1shape), (7,))) isa AbstractArray{<:AbstractVector{<:Real},1}
        @test size(rand(_mvnormalrd(a1shape), 7)) == (7,)

        @test @inferred(rand(_mvnormalrd(a2shape))) isa AbstractMatrix{<:Real}
        @test @inferred(rand(_mvnormalrd(a2shape), ())) isa AbstractMatrix{<:Real}
        @test @inferred(rand(_mvnormalrd(a2shape), 7)) isa AbstractArray{<:AbstractMatrix{<:Real},1}
        @test size(rand(_mvnormalrd(a2shape), 7)) == (7,)

        @test @inferred(rand(_mvnormalrd(ntshape))) isa NamedTuple{names}
        @test @inferred(rand(_mvnormalrd(ntshape), ())) isa ShapedAsNT{<:NamedTuple{names}}
        @test @inferred(rand(_mvnormalrd(ntshape), 7)) isa ShapedAsNTArray{<:NamedTuple{names},1}
        @test size(rand(_mvnormalrd(ntshape), 7)) == (7,)
    end

    @testset "stats functions" begin
        @test @inferred(mean(_mvnormalrd(scshape))) ≈ 2
        @test @inferred(mode(_mvnormalrd(scshape))) ≈ 2
        @test @inferred(var(_mvnormalrd(scshape))) ≈ 3

        @test @inferred(mean(_mvnormalrd(a1shape))) ≈ fill(2, 3)
        @test @inferred(mode(_mvnormalrd(a1shape))) ≈ fill(2, 3)
        @test @inferred(var(_mvnormalrd(a1shape))) ≈ fill(3, 3)
        @test @inferred(cov(_mvnormalrd(a1shape))) ≈ Diagonal(fill(3.0, 3))

        @test @inferred(mean(_mvnormalrd(a2shape))) ≈ fill(2, 2, 3)
        @test @inferred(mode(_mvnormalrd(a2shape))) ≈ fill(2, 2, 3)
        @test @inferred(var(_mvnormalrd(a2shape))) ≈ fill(3, 2, 3)

        @test @inferred(mean(_mvnormalrd(ntshape))) == (a = [2.0 2.0 2.0; 2.0 2.0 2.0], b = 2.0, c = 4.2, x = [11 21; 12 22], y = [2.0, 2.0, 2.0, 2.0])
        @test @inferred(mode(_mvnormalrd(ntshape))) == (a = [2.0 2.0 2.0; 2.0 2.0 2.0], b = 2.0, c = 4.2, x = [11 21; 12 22], y = [2.0, 2.0, 2.0, 2.0])
        @test @inferred(var(_mvnormalrd(ntshape))) == (a = [3.0 3.0 3.0; 3.0 3.0 3.0], b = 3.0, c = 0.0, x = [0 0; 0 0], y = [3.0, 3.0, 3.0, 3.0])
    end
end
