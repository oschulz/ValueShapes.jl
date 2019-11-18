# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test


@testset "functions" begin
    shape = NamedTupleShape(
        a = ArrayShape{Real}(3,2),
        b = ArrayShape{Real}(2)
    )

    x_flat = rand(@inferred totalndof(shape))
    x = @inferred shape(x_flat)

    f(x) = sum(x.a * x.b)

    @test @inferred(shape >> f) isa ValueShapes.FuncWithVarShape
    @test @inferred(varshape((shape >> f))) == shape
    @test @inferred(vardof((shape >> f))) == totalndof(shape)

    fws = shape >> f
    @test f(x) == @inferred(fws(x))
    @test f(x) == @inferred(fws(x_flat))
end
