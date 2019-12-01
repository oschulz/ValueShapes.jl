# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions, IntervalSets

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

    # ToDo: Add more tests
end
