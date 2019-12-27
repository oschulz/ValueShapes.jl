# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions
using Random


@testset "const_value_dist" begin
    uni_dist = Normal(1, 0)
    cvd = ConstValueDist(Int64(42))
    shape = varshape(cvd)

    @test @inferred insupport(cvd, 42) == true
    @test @inferred insupport(cvd, 41.999) == false
    @test @inferred insupport(cvd, 42.999) == false
    @test @inferred totalndof(shape) == 0
    @test @inferred size(cvd) == ()
    @test @inferred length(cvd) == 1
    
    # These two lines don't reach the expected section of code
    @test @inferred minimum(cvd.value) == 42
    @test @inferred maximum(cvd.value) == 42
    @test @inferred maximum(pdf(cvd, -50:50)) == Inf
    
    @test @inferred pdf(cvd, 42) == Inf
    
    @test @inferred cdf(cvd, 41.999) == 0
    @test @inferred cdf(cvd, 42) == 1
    @test @inferred cdf(cvd, Inf) == 1

    @test @inferred mode(cvd) == 42

    @test @inferred rand(cvd, 10) == fill(42, 10)

    @test @inferred logpdf(cvd, -1) == -Inf 
    @test @inferred logpdf(cvd, 41.999) == -Inf 
    @test @inferred logpdf(cvd, 42) == Inf 
    @test @inferred logpdf(cvd, 43.999) == -Inf 
    
    @test @inferred eltype(cvd) == Int64
end
