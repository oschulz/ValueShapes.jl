# This file is a part of ParameterShapes.jl, licensed under the MIT License (MIT).

using ParameterShapes
using Test


@testset "npargs" begin
    @test @macroexpand(@npargs a::Integer, b, c::AbstractVector) == :((a, b, c)::NamedTuple{(:a, :b, :c), <:Tuple{Integer, Any, AbstractVector}})

    foo(@npargs a::Integer, b, c::AbstractVector) = a * b * c

    @test foo((a = 42, b = 4.2, c = [2, 5])) ≈ [352.8, 882.0]

    @test_throws MethodError foo((a = 4.2, b = 42, c = 25)) ≈ [352.8, 882.0]
end
