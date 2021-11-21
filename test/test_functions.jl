# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using LinearAlgebra


@testset "functions" begin
    input_shape = NamedTupleShape(
        ShapedAsNT,
        a = ArrayShape{Real}(3,2),
        b = ArrayShape{Real}(2)
    )

    output_shape = NamedTupleShape(
        ShapedAsNT,
        x = ScalarShape{Real}(),
        y = ArrayShape{Real}(3)
    )


    @testset "UnshapedFunction" begin
        f(x) = (x = norm(x.a)^2 + norm(x.b)^2, y = vec(sum(x.a, dims = 2)))
        ux = [2, 7, 4, 6, 8, 5, 3, 1]
        x = input_shape(ux)[]
        y = f(x)
        uy = unshaped(y, output_shape)
        @test @inferred(unshaped(f, input_shape)(ux)) == y
        @test @inferred(unshaped(f, input_shape, nothing)(ux)) == y
        @test @inferred(unshaped(f, input_shape, output_shape)(ux)) == uy
        @test @inferred(unshaped(f, nothing, output_shape)(x)) == uy
        @test @inferred(unshaped(f, nothing)(x)) == y
        @test @inferred(unshaped(f, nothing, nothing)(x)) == y
    end


    @testset "FuncWithVarShape" begin
        x_flat = rand(@inferred totalndof(input_shape))
        x = @inferred stripscalar(input_shape(x_flat))

        g(x) = sum(x.a * x.b)

        @test @inferred(input_shape >> g) isa ValueShapes.FuncWithVarShape
        @test @inferred(varshape((input_shape >> g))) == input_shape
        @test @inferred(vardof((input_shape >> g))) == totalndof(input_shape)

        fws = input_shape >> g
        @test g(x) == @inferred(fws(x))
        @test g(x) == @inferred(fws(x_flat))
    end
end
