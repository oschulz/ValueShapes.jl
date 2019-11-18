# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions


@testset "distributions" begin
    @test @inferred(varshape(Normal())) == ScalarShape{Real}()
    @test @inferred(varshape(MvNormal([2. 1.; 1. 3.]))) == ArrayShape{Real}(2)
    @test @inferred(varshape(MatrixBeta(4, 6, 6))) == ArrayShape{Real}(4, 4)

    @test @inferred(vardof(MvNormal([2. 1.; 1. 3.]))) == 2
end
