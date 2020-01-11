# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Statistics, StatsBase, Distributions, IntervalSets

@testset "NamedTupleDist" begin
    dist = @inferred NamedTupleDist(
        a = 5, b = Normal(),
        c = -4..5,
        d = MvNormal([1.2 0.5; 0.5 2.1]),
        x = [3 4; 2 5],
        e = [Normal(1.1, 0.2)]
    )

    @test typeof(@inferred varshape(dist)) <: NamedTupleShape

    shape = varshape(dist)

    @test (@inferred logpdf(dist, [0.2, -0.4, 0.3, -0.5, 0.9])) == logpdf(Normal(), 0.2) + logpdf(Uniform(-4, 5), -0.4) + logpdf(MvNormal([1.2 0.5; 0.5 2.1]), [0.3, -0.5]) + logpdf(Normal(1.1, 0.2), 0.9)

    @test (@inferred logpdf(dist, shape([0.2, -0.4, 0.3, -0.5, 0.9])[])) == logpdf(Normal(), 0.2) + logpdf(Uniform(-4, 5), -0.4) + logpdf(MvNormal([1.2 0.5; 0.5 2.1]), [0.3, -0.5]) + logpdf(Normal(1.1, 0.2), 0.9)

    @test (@inferred mode(dist)) == [0.0, 0.5, 0.0, 0.0, 1.1]

    @test begin
        ref_cov = 
            [1.0  0.0   0.0  0.0 0.0;
             0.0  6.75  0.0  0.0 0.0;
             0.0  0.0   1.2  0.5 0.0;
             0.0  0.0   0.5  2.1 0.0; 
             0.0  0.0   0.0  0.0 0.04 ]

        @static if VERSION >= v"1.2"
            (@inferred cov(dist)) ≈ ref_cov
        else
            (cov(dist)) ≈ ref_cov
        end
    end

    dist_h = @inferred NamedTupleDist(
        h1 = fit(Histogram, randn(10^5)),
        h2 = fit(Histogram, (2 * randn(10^5), 3 * randn(10^5)))
)
    @test isapprox(std(@inferred(rand(dist_h, 10^5)), dims = 2, corrected = true), [1, 2, 3], rtol = 0.1)
    
    propnames = propertynames(dist, true)
    @test propnames == (:a, :b, :c, :d, :x, :e, :_internal_distributions, :_internal_shapes)
    @test @inferred(keys(dist)) == propertynames(dist, false)
    internaldists = getproperty(dist, :_internal_distributions)
    internalshapes = getproperty(dist, :_internal_shapes) 
    for i in 1:length(internaldists)
        @test typeof(getindex(internaldists, i)) == typeof(getproperty(dist, keys(dist)[i]))
    end
    for key in keys(internalshapes)
        @test getproperty(getproperty(internalshapes, key), :shape) == valshape(getproperty(internalshapes, key))
    end


    # ToDo: Add more tests
end
