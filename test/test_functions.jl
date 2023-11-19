# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using LinearAlgebra


@testset "functions" begin
    f(x) = [x]
    ValueShapes.resultshape(::typeof(f), ::Real) = ArrayShape{Real}(1)

    @test @inferred(resultshape(f, 42)) >= valshape(f(42))
end
