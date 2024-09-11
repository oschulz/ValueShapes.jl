# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ElasticArrays
using ArraysOfArrays
using FillArrays
using InverseFunctions, ChangesOfVariables
using ChainRulesCore: rrule, NoTangent
import TypedTables
import Dates


@testset "abstract_value_shape" begin
    @testset "default_datatype" begin
        @test @inferred(ValueShapes.default_datatype(Integer)) == Int
        @test @inferred(ValueShapes.default_datatype(AbstractFloat)) == Float64
        @test @inferred(ValueShapes.default_datatype(Real)) == Float64
        @test @inferred(ValueShapes.default_datatype(Complex)) == Complex{Float64}

        @test @inferred(elshape(Complex(1.0, 2.0))) == ScalarShape{Complex{Float64}}()
        @test @inferred(elshape([[3, 5], [3, 2]])) == ArrayShape{Int,1}((2,))

        @test ValueShapes.stripscalar(Ref(ScalarShape{Real})) == ScalarShape{Real}

        # @test Vector{Real}(undef, ArrayShape{Real}((2,1))) ==  # weird typing going on with default_unshaped_eltype

        arrshape = ArrayShape{Real, 2}((2,3))
        v = Vector{Real}(undef, arrshape)
        @test @inferred(length(v)) == 6
        @test @inferred(size(v)) == (6,)

        data1 = [1;2;3;4;7;8;9]
        scalarshape = ScalarShape{Real}()
        ntshape = NamedTupleShape(a=arrshape, b=scalarshape)
        shapedasnt = ntshape(data1)
        @test stripscalar(Ref(shapedasnt)) == Ref(shapedasnt)[]

        @test_throws ArgumentError Broadcast.broadcastable(ntshape)

        named_shapes = (
            a = ArrayShape{Real}(2, 3),
            b = ScalarShape{Real}(),
            c = ConstValueShape(4.2),
            x = ConstValueShape([11 21; 12 22]),
            y = ArrayShape{Real}(4)
        )
        shape = NamedTupleShape(;named_shapes...)
        sntshape = NamedTupleShape(ShapedAsNT; named_shapes...)
        data2 = VectorOfSimilarVectors(reshape(collect(1:22), 11, 2))
        @test_throws ArgumentError ValueShapes._checkcompat_inner(ntshape, data2)
        @test ValueShapes._checkcompat_inner(shape, data2) == nothing

        let vs = sntshape, x = rand(totalndof(vs)), xs = nestedview(rand(totalndof(vs), 5))
            vs_jacobian(f, x) = 1

            InverseFunctions.test_inverse(vs, x)
            ChangesOfVariables.test_with_logabsdet_jacobian(vs, x, vs_jacobian)
            ChangesOfVariables.test_with_logabsdet_jacobian(inverse(vs), vs(x), vs_jacobian)

            bc_vs = Base.Fix1(broadcast, vs)
            bc_unshaped = Base.Fix1(broadcast, unshaped)
            InverseFunctions.test_inverse(bc_vs, xs)
            @test with_logabsdet_jacobian(bc_vs, xs)[1] isa ShapedAsNTArray
            ChangesOfVariables.test_with_logabsdet_jacobian(bc_vs, xs, vs_jacobian)
            @test with_logabsdet_jacobian(inverse(bc_vs), vs.(xs))[1] isa ArrayOfSimilarArrays
            ChangesOfVariables.test_with_logabsdet_jacobian(inverse(bc_vs), vs.(xs), vs_jacobian)
            @test with_logabsdet_jacobian(bc_unshaped, vs.(xs))[1] isa ArrayOfSimilarArrays
            ChangesOfVariables.test_with_logabsdet_jacobian(bc_unshaped, vs.(xs), vs_jacobian)

            for f in [vs, inverse(vs), bc_vs, inverse(bc_vs), unshaped]
                @test @inferred(identity ∘ f) === f
                @test @inferred(f ∘ identity) === f
            end
        end

        @test @inferred(unshaped(4.2)) isa Fill{Float64,1}
        @test unshaped(4.2) == [4.2]
        @test @inferred(unshaped(view([4.2], 1))) isa SubArray{Float64,1,Vector{Float64}}
        @test unshaped(view([4.2], 1)) == [4.2]
        @test @inferred(unshaped(Array(view([4.2], 1)))) isa SubArray{Float64,1,Vector{Float64}}
        @test unshaped(Array(view([4.2], 1))) == [4.2]
        let x = rand(15)
            @test @inferred(unshaped(x)) === x
            @test @inferred(unshaped(Base.ReshapedArray(x, (3, 5), ()))) === x

            @test @inferred(broadcast(unshaped, x)) isa ArrayOfSimilarArrays{Float64,1,1,2,<:Base.ReshapedArray}
            @test broadcast(unshaped, x) == nestedview(reshape(x, 1, 15))
        end

        let A = rand(1,15)
            @test @inferred(broadcast(unshaped, view(A, 1, :))) isa ArrayOfSimilarArrays{Float64,1,1,2,<:SubArray}
            @test broadcast(unshaped, view(A, 1, :)) == nestedview(A)
        end
    end

    @testset "realnumtype" begin
        a = 4.2f0
        b = Complex(a, a)
        A = fill(fill(rand(Float32, 5), 10), 5)
        B = fill(rand(Float32, 5), 10)
        C = ArrayOfSimilarArrays(B)
        nt = (a = 4.2f0, b = 42)
        tpl = (4.2f0, Float16(2.3))
        deepnt = (a = a, b = b, A = A, B = B, C = C, nt = nt, tpl = tpl)
        for x in [a, A, B, C, nt, tpl, deepnt]
            @test @inferred (realnumtype(typeof(x))) == Float32
        end

        @test @inferred(realnumtype(typeof(Dates.TWENTYFOURHOUR))) <: Integer
        for x in [nothing, missing, (), :foo, "foo"]
            @test @inferred(realnumtype(typeof(x))) == Bool
        end
    end

    @testset "value shape comparison" begin
        for (T, U) in [(Real, Float64), (Float64, Real), (Real, Real), (Float64, Float64)]
            @test @inferred(ScalarShape{T}() <= ScalarShape{U}())[1] == (T <: U)
            @test @inferred(rrule(Base.:(<=), ScalarShape{T}(), ScalarShape{U}()))[1] == (T <: U)
            @test @inferred(rrule(Base.:(<=), ScalarShape{T}(), ScalarShape{U}()))[2](T <: U) == (NoTangent(), NoTangent(), NoTangent())

            @test @inferred(ScalarShape{T}() >= ScalarShape{U}())[1] == (T >: U)
            @test @inferred(rrule(Base.:(>=), ScalarShape{T}(), ScalarShape{U}()))[1] == (T >: U)
            @test @inferred(rrule(Base.:(>=), ScalarShape{T}(), ScalarShape{U}()))[2](T >: U) == (NoTangent(), NoTangent(), NoTangent())

            @test @inferred(ScalarShape{T}() == ScalarShape{U}())[1] == (T == U)
            @test @inferred(rrule(Base.:(==), ScalarShape{T}(), ScalarShape{U}()))[1] == (T == U)
            @test @inferred(rrule(Base.:(==), ScalarShape{T}(), ScalarShape{U}()))[2](T == U) == (NoTangent(), NoTangent(), NoTangent())
        end
    end
end
