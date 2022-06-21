# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using LinearAlgebra, Random, Statistics
using StatsBase, Distributions, ArraysOfArrays


@testset "reshaped_dist" begin
    scshape = ScalarShape{Real}()

    a1shape = ArrayShape{Real}(3)
    a2shape = ArrayShape{Real}(2, 3)

    ntshape = NamedTupleShape(
        ShapedAsNT,
        a = ArrayShape{Real}(2, 3),
        b = ScalarShape{Real}(),
        c = ConstValueShape(4.2),
        x = ConstValueShape([11 21; 12 22]),
        y = ArrayShape{Real}(4)
    )

    nms = (:a, :b, :c, :x, :y)
    
    _mvndist(n::Integer) = MvNormal(fill(2.0, n), Diagonal(fill(3.0, n)))
    _mvnormalrd(shape::AbstractValueShape) = ReshapedDist(_mvndist(totalndof(shape)), shape)
    _mvnormalrd2(shape::AbstractValueShape) = shape(_mvndist(totalndof(shape)))

    @testset "ctors" begin
        @test @inferred(_mvnormalrd(scshape)) isa Distribution{Univariate, Continuous}
        @test @inferred(_mvnormalrd(a1shape)) isa Distribution{Multivariate, Continuous}
        @test @inferred(_mvnormalrd(a2shape)) isa Distribution{Matrixvariate, Continuous}
        @test @inferred(_mvnormalrd(ntshape)) isa Distribution{ValueShapes.NamedTupleVariate{nms}, Continuous}

        @test_throws ArgumentError ReshapedDist(MvNormal(Diagonal(fill(3.0, 2))), a2shape)

        @test @inferred(_mvnormalrd2(scshape)) isa ReshapedDist{Univariate, Continuous}
        @test @inferred(_mvnormalrd2(a1shape)) isa MvNormal
        @test @inferred(_mvnormalrd2(a2shape)) isa MatrixReshaped
        @test @inferred(_mvnormalrd2(ntshape)) isa ReshapedDist{ValueShapes.NamedTupleVariate{nms}, Continuous}

        @inferred(varshape(_mvnormalrd2(ntshape))) == ntshape
        @inferred(unshaped(_mvnormalrd2(ntshape))) == MvNormal(fill(2.0, totalndof(ntshape)), Diagonal(fill(3.0, totalndof(ntshape))))
    end

    @testset "rand" begin
        @test @inferred(rand(_mvnormalrd(scshape))) isa Real
        @test @inferred(rand(_mvnormalrd(scshape), ())) isa AbstractArray{<:Real,0}
        @test @inferred(rand(_mvnormalrd(scshape), 7)) isa AbstractArray{<:Real,1}
        @test @inferred(rand(_mvnormalrd(scshape), (7,))) isa AbstractArray{<:Real,1}
        @test size(rand(_mvnormalrd(scshape), 7)) == (7,)

        let d = _mvndist(totalndof(a1shape))
            @test @inferred(a1shape(d)) === d
        end
        @test @inferred(rand(_mvnormalrd(a1shape))) isa AbstractVector{<:Real}
        @test @inferred(rand(_mvnormalrd(a1shape), ())) isa AbstractArray{<:AbstractVector{<:Real},0}
        @test @inferred(rand(_mvnormalrd(a1shape), 7)) isa AbstractArray{<:Real,2}
        @test @inferred(rand(_mvnormalrd(a1shape), (7,))) isa AbstractArray{<:AbstractVector{<:Real},1}
        @test size(rand(_mvnormalrd(a1shape), 7)) == (3, 7)
        @test size(rand(_mvnormalrd(a1shape), (7,))) == (7,)

        let d = _mvndist(totalndof(a2shape))
            @test @inferred(a2shape(d)) isa MatrixReshaped
            @test unshaped(a2shape(d)) === d
        end
        @test @inferred(rand(_mvnormalrd(a2shape))) isa AbstractMatrix{<:Real}
        @test @inferred(rand(_mvnormalrd(a2shape), 7)) isa AbstractArray{<:AbstractMatrix{<:Real},1}
        @test size(rand(_mvnormalrd(a2shape), 7)) == (7,)

        @test @inferred(rand(_mvnormalrd(ntshape))) isa ShapedAsNT{nms}
        @test @inferred(rand(_mvnormalrd(ntshape), ())) isa ShapedAsNTArray{<:ShapedAsNT{nms},0}
        @test @inferred(rand(_mvnormalrd(ntshape), 7)) isa ShapedAsNTArray{<:ShapedAsNT{nms},1}
        @test size(rand(_mvnormalrd(ntshape), 7)) == (7,)
        let X = rand(_mvnormalrd(ntshape), 7)
            @test @inferred(rand!(_mvnormalrd(ntshape), view(X, 1))) == view(X, 1)
            @test @inferred(rand!(_mvnormalrd(ntshape), X)) === X
        end
    end

    @testset "stats functions" begin
        @test @inferred(mean(_mvnormalrd(scshape))) ≈ 2
        @test @inferred(mode(_mvnormalrd(scshape))) ≈ 2
        @test @inferred(var(_mvnormalrd(scshape))) ≈ 3
        let rd = _mvnormalrd(scshape), ux = rand(unshaped(rd)), vs = varshape(rd)
            @test @inferred(pdf(rd, vs(ux)[])) == pdf(unshaped(rd), ux)
            @test @inferred(pdf(rd, vs(ux))) == pdf(unshaped(rd), ux)
            @test @inferred(logpdf(rd, vs(ux)[])) == logpdf(unshaped(rd), ux)
            @test @inferred(logpdf(rd, vs(ux))) == logpdf(unshaped(rd), ux)
            @test @inferred(Distributions.insupport(rd, vs(ux)[])) == true
        end

        @test @inferred(mean(_mvnormalrd(a1shape))) ≈ fill(2, 3)
        @test @inferred(mode(_mvnormalrd(a1shape))) ≈ fill(2, 3)
        @test @inferred(var(_mvnormalrd(a1shape))) ≈ fill(3, 3)
        @test @inferred(cov(_mvnormalrd(a1shape))) ≈ Diagonal(fill(3.0, 3))
        let rd = _mvnormalrd(a1shape), ux = rand(unshaped(rd)), vs = varshape(rd)
            @test @inferred(pdf(rd, vs(ux))) == pdf(unshaped(rd), ux)
            @test @inferred(logpdf(rd, vs(ux))) == logpdf(unshaped(rd), ux)
            @test @inferred(Distributions.insupport(rd, vs(ux))) == true
        end

        @test @inferred(mean(_mvnormalrd(a2shape))) ≈ fill(2, 2, 3)
        @test @inferred(mode(_mvnormalrd(a2shape))) ≈ fill(2, 2, 3)
        @test @inferred(var(_mvnormalrd(a2shape))) ≈ fill(3, 2, 3)
        let rd = _mvnormalrd(a2shape), ux = rand(unshaped(rd)), vs = varshape(rd)
            @test @inferred(pdf(rd, vs(ux))) == pdf(unshaped(rd), ux)
            @test @inferred(logpdf(rd, vs(ux))) == logpdf(unshaped(rd), ux)
            @test @inferred(Distributions.insupport(rd, vs(ux))) == true
        end

        @test @inferred(mean(_mvnormalrd(ntshape)))[] == (a = [2.0 2.0 2.0; 2.0 2.0 2.0], b = 2.0, c = 4.2, x = [11 21; 12 22], y = [2.0, 2.0, 2.0, 2.0])
        @test @inferred(mode(_mvnormalrd(ntshape)))[] == (a = [2.0 2.0 2.0; 2.0 2.0 2.0], b = 2.0, c = 4.2, x = [11 21; 12 22], y = [2.0, 2.0, 2.0, 2.0])
        @test @inferred(var(_mvnormalrd(ntshape)))[] == (a = [3.0 3.0 3.0; 3.0 3.0 3.0], b = 3.0, c = 0.0, x = [0 0; 0 0], y = [3.0, 3.0, 3.0, 3.0])
        let rd = _mvnormalrd(ntshape), ux = rand(unshaped(rd)), vs = varshape(rd)
            @test @inferred(pdf(rd, vs(ux)[])) == pdf(unshaped(rd), ux)
            @test @inferred(pdf(rd, vs(ux))) == pdf(unshaped(rd), ux)
            @test @inferred(logpdf(rd, vs(ux)[])) == logpdf(unshaped(rd), ux)
            @test @inferred(logpdf(rd, vs(ux))) == logpdf(unshaped(rd), ux)
            @test @inferred(Distributions.insupport(rd, vs(ux)[])) == true
            @test @inferred(Distributions.insupport(rd, vs(ux))) == true
        end

        let rd = ReshapedDist(_mvndist(5), ArrayShape{Real}(5)), X = rand(rd, 10), Xn = nestedview(X)
            @test @inferred(Distributions._pdf(rd, Xn[1])) == pdf(rd, Xn[1])
            @test @inferred(pdf(rd, X)) == pdf.(Ref(rd), Xn)
        
            @test @inferred(Distributions._logpdf(rd, Xn[1])) == logpdf(rd, Xn[1])
            @test @inferred(logpdf(rd, X)) == logpdf.(Ref(rd), Xn)
        end    
    end
end
