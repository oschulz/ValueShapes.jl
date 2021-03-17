# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions
using Random


@testset "const_value_dist" begin
    @test @inferred(ConstValueDist(4.2)) isa Distribution{Univariate}
    @test @inferred(ConstValueDist([1, 2, 3])) isa Distribution{Multivariate}
    @test @inferred(ConstValueDist([1 2; 3 4])) isa Distribution{Matrixvariate}

    @test @inferred(length(ConstValueDist([4.2]))) == 1
    @test @inferred(length(ConstValueDist([1, 2, 3]))) == 3
    @test @inferred(length(ConstValueDist([1 2; 3 4]))) == 4

    @test @inferred(pdf(ConstValueDist(4.2), 4.2)) == Inf
    @test @inferred(logpdf(ConstValueDist(4.2), 4.2)) == Inf
    @test @inferred(pdf(ConstValueDist(4.2), 3.7)) == 0
    @test @inferred(logpdf(ConstValueDist(4.2), 3.7)) == -Inf
    @test @inferred(pdf(ConstValueDist(4.2), [4.2, 3.7])) == [Inf, 0.0]
    @test @inferred(logpdf(ConstValueDist(4.2), [4.2, 3.7])) == [Inf, -Inf]

    @test @inferred(pdf(ConstValueDist([1, 2, 3]), [1, 2, 3])) == Inf
    @test @inferred(logpdf(ConstValueDist([1, 2, 3]), [1, 2, 3])) == Inf
    @test @inferred(pdf(ConstValueDist([1, 2, 3]), [2, 3, 4])) == 0
    @test @inferred(logpdf(ConstValueDist([1, 2, 3]), [2, 3, 4])) == -Inf
    @test (pdf(ConstValueDist([1, 2, 3]), hcat([1, 2, 3], [2, 3, 4]))) == [Inf, 0.0]
    @test (logpdf(ConstValueDist([1, 2, 3]), hcat([1, 2, 3], [2, 3, 4]))) == [Inf, -Inf]

    @test @inferred(pdf(ConstValueDist([1 2; 3 4]), [1 2; 3 4])) == Inf
    @test @inferred(logpdf(ConstValueDist([1 2; 3 4]), [1 2; 3 4])) == Inf
    @test @inferred(pdf(ConstValueDist([1 2; 3 4]), [4 5; 6 7])) == 0
    @test @inferred(logpdf(ConstValueDist([1 2; 3 4]), [4 5; 6 7])) == -Inf

    @test @inferred(insupport(ConstValueDist(4.2), 4.2)) == true
    @test @inferred(insupport(ConstValueDist(4.2), 4.19)) == false

    @test @inferred(insupport(ConstValueDist([1, 2, 3]), [1, 2, 3])) == true
    @test @inferred(insupport(ConstValueDist([1, 2, 3]), [2, 3, 4])) == false

    @test @inferred(insupport(ConstValueDist([1 2; 3 4]), [1 2; 3 4])) == true
    @test @inferred(insupport(ConstValueDist([1 2; 3 4]), [4 5; 6 7])) == false


    @test @inferred(rand(ConstValueDist(4.2))) == 4.2
    @test @inferred(rand(ConstValueDist(4.2), 3)) == [4.2, 4.2, 4.2]

    @test @inferred(rand(ConstValueDist([1, 2, 3]))) == [1, 2, 3]
    @test @inferred(rand(ConstValueDist([1, 2, 3]), 2)) == hcat([1, 2, 3], [1, 2, 3])

    @test @inferred(rand(ConstValueDist([1 2; 3 4]))) == [1 2; 3 4]
    @test @inferred(rand(ConstValueDist([1 2; 3 4]), 2)) == [[1 2; 3 4], [1 2; 3 4]]


    univariate_cvd = @inferred(ConstValueDist(Int64(42)))

    shape = varshape(univariate_cvd)

    @test @inferred(totalndof(shape)) == 0
    @test @inferred(size(univariate_cvd)) == ()
    @test @inferred(length(univariate_cvd)) == 1

    @test @inferred(minimum(univariate_cvd)) == 42
    @test @inferred(maximum(univariate_cvd)) == 42

    @test @inferred(pdf(univariate_cvd, 42)) == Inf

    @test @inferred(cdf(univariate_cvd, 41.999)) == 0
    @test @inferred(cdf(univariate_cvd, 42)) == 1
    @test @inferred(cdf(univariate_cvd, Inf)) == 1

    @test @inferred(mode(univariate_cvd)) == 42

    @test @inferred(eltype(univariate_cvd)) == Int64

 
    univariate_value = 1.0
    univariate_cvd = @inferred(ConstValueDist((a=univariate_value,)))
    univariate_samples = @inferred(rand(univariate_cvd, 10^3))
    univariate_samples_tofill = @inferred(similar(univariate_samples))

    @test @inferred(pdf(univariate_cvd, (a=univariate_value,))) == Inf
    @test @inferred(pdf(univariate_cvd, (a=univariate_value - eps(eltype(univariate_cvd)),))) == 0
    @test @inferred(pdf(univariate_cvd, (a=univariate_value + eps(eltype(univariate_cvd)),))) == 0

    @test @inferred(logpdf(univariate_cvd, (a=univariate_value,))) == Inf
    @test @inferred(logpdf(univariate_cvd, (a=univariate_value - eps(eltype(univariate_cvd)),))) == -Inf
    @test @inferred(logpdf(univariate_cvd, (a=univariate_value + eps(eltype(univariate_cvd)),))) == -Inf

    @test @inferred(insupport(univariate_cvd, (a=univariate_value,))) == true
    @test @inferred(insupport(univariate_cvd, (a=univariate_value - eps(eltype(univariate_cvd)),))) == false
    @test @inferred(insupport(univariate_cvd, (a=univariate_value + eps(eltype(univariate_cvd)),))) == false
  
    @test univariate_samples == fill((a=univariate_value,), 10^3)
    @test @inferred(rand!(univariate_cvd, univariate_samples_tofill)) == univariate_samples
end
