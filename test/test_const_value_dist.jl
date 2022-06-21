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

    @test @inferred(pdf(ConstValueDist(4.2), 4.2)) == 1
    @test @inferred(logpdf(ConstValueDist(4.2), 4.2)) == 0
    @test @inferred(pdf(ConstValueDist(4.2), 3.7)) == 0
    @test @inferred(logpdf(ConstValueDist(4.2), 3.7)) == -Inf
    @test @inferred(broadcast(pdf, ConstValueDist(4.2), [4.2, 3.7])) == [1, 0]
    @test @inferred(broadcast(logpdf, ConstValueDist(4.2), [4.2, 3.7])) == [0, -Inf]

    @test @inferred(pdf(ConstValueDist([1, 2, 3]), [1, 2, 3])) == 1
    @test @inferred(logpdf(ConstValueDist([1, 2, 3]), [1, 2, 3])) == 0
    @test @inferred(pdf(ConstValueDist([1, 2, 3]), [2, 3, 4])) == 0
    @test @inferred(logpdf(ConstValueDist([1, 2, 3]), [2, 3, 4])) == -Inf
    @test (pdf(ConstValueDist([1, 2, 3]), hcat([1, 2, 3], [2, 3, 4]))) == [1, 0]
    @test (logpdf(ConstValueDist([1, 2, 3]), hcat([1, 2, 3], [2, 3, 4]))) == [0, -Inf]

    @test @inferred(pdf(ConstValueDist([1 2; 3 4]), [1 2; 3 4])) == 1
    @test @inferred(logpdf(ConstValueDist([1 2; 3 4]), [1 2; 3 4])) == 0
    @test @inferred(pdf(ConstValueDist([1 2; 3 4]), [4 5; 6 7])) == 0
    @test @inferred(logpdf(ConstValueDist([1 2; 3 4]), [4 5; 6 7])) == -Inf

    @test @inferred(Distributions.insupport(ConstValueDist(4.2), 4.2)) == true
    @test @inferred(Distributions.insupport(ConstValueDist(4.2), 4.19)) == false

    @test @inferred(Distributions.insupport(ConstValueDist([1, 2, 3]), [1, 2, 3])) == true
    @test @inferred(Distributions.insupport(ConstValueDist([1, 2, 3]), [2, 3, 4])) == false

    @test @inferred(Distributions.insupport(ConstValueDist([1 2; 3 4]), [1 2; 3 4])) == true
    @test @inferred(Distributions.insupport(ConstValueDist([1 2; 3 4]), [4 5; 6 7])) == false


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

    @test @inferred(pdf(univariate_cvd, 42)) == 1

    @test @inferred(cdf(univariate_cvd, 41.999)) == 0
    @test @inferred(cdf(univariate_cvd, 42)) == 1
    @test @inferred(cdf(univariate_cvd, Inf)) == 1

    @test @inferred(mean(univariate_cvd)) == 42

    @test @inferred(mode(univariate_cvd)) == 42

    @test @inferred(eltype(univariate_cvd)) == Int64
    
    μ1, μ2 = rand(2)
    cvd_from_named_tuple = @inferred(ConstValueDist((a=μ1, b=μ2)))

    @test @inferred(log(@inferred(pdf(cvd_from_named_tuple, (a=μ1, b=μ2))))) == @inferred(logpdf(cvd_from_named_tuple, (a=μ1, b=μ2))) == 0

    @test @inferred(Distributions.insupport(cvd_from_named_tuple, (a=μ1, b=μ2))) == true
    @test @inferred(Distributions.insupport(cvd_from_named_tuple, (a=μ1+eps(μ1), b=μ2+eps(μ2)))) == false
    @test @inferred(Distributions.insupport(cvd_from_named_tuple, (a=μ1+eps(μ1), b=μ2))) == false
    @test @inferred(Distributions.insupport(cvd_from_named_tuple, (a=μ1, b=μ2+eps(μ2)))) == false

    n_samples = 100
    samples = @inferred(rand(cvd_from_named_tuple, n_samples))
    emptied_samples = @inferred(similar(samples))

    @test emptied_samples != samples
    @test samples == @inferred(fill((a=μ1, b=μ2), n_samples))
    @test @inferred(rand!(cvd_from_named_tuple, emptied_samples)) == samples
    @test samples == emptied_samples
end
