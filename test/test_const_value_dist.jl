# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions
using Random


@testset "const_value_dist" begin
    univariate_cvd = ConstValueDist(Int64(42))
    v = broadcast(Int64, [1,2, 3])
    multivariate_cvd = ConstValueDist(v)
    @test @inferred(length(multivariate_cvd)) == length(v)
    
    shape = varshape(univariate_cvd)

    @test @inferred(insupport(univariate_cvd, 42)) == true
    @test @inferred(insupport(univariate_cvd, 41.999)) == false
    @test @inferred(insupport(univariate_cvd, 42.999)) == false
    @test @inferred(totalndof(shape)) == 0
    @test @inferred(size(univariate_cvd)) == ()
    @test @inferred(length(univariate_cvd)) == 1
    
    @test @inferred(minimum(univariate_cvd)) == 42
    @test @inferred(maximum(univariate_cvd)) == 42
    @test @inferred(maximum(@inferred(pdf(univariate_cvd, -50:50)))) == Inf
    
    @test @inferred(pdf(univariate_cvd, 42)) == Inf
    
    @test @inferred(cdf(univariate_cvd, 41.999)) == 0
    @test @inferred(cdf(univariate_cvd, 42)) == 1
    @test @inferred(cdf(univariate_cvd, Inf)) == 1

    @test @inferred(mode(univariate_cvd)) == 42

    @test @inferred(rand(univariate_cvd, 10)) == fill(42, 10)

    @test @inferred(logpdf(univariate_cvd, -1)) == -Inf
    @test @inferred(logpdf(univariate_cvd, 41.999)) == -Inf
    @test @inferred(logpdf(univariate_cvd, 42)) == Inf
    @test @inferred(logpdf(univariate_cvd, 43.999)) == -Inf 
    
    @test @inferred(eltype(univariate_cvd)) == Int64
end
