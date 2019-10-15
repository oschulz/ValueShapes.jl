# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions


@testset "distributions_support" begin
    @test @inferred(valshape(Normal())) == ScalarShape{Real}()
    @test @inferred(valshape(MvNormal([2. 1.; 1. 3.]))) == ArrayShape{Real}(2)
    @test @inferred(valshape(MatrixBeta(4, 6, 6))) == ArrayShape{Real}(4, 4)
end
