# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using StatsBase, IntervalSets, ValueShapes, ArraysOfArrays
using MeasureBase, Distributions, DistributionMeasures

include("testutils.jl")


@testset "hierarchial_distribution" begin
    let
        primary_dist = NamedTupleDist(
            foo = LogNormal(1, 0.3),
            bar = Normal(2.0, 1.0)
        )

        snt_primary_dist = NamedTupleDist(
            ShapedAsNT,
            foo = LogNormal(1, 0.3),
            bar = Normal(2.0, 1.0)
        )

        f = v -> NamedTupleDist(baz = fill(Normal(v.bar, v.foo), 3))

        @test @inferred(HierarchicalDist(f, primary_dist)) isa HierarchicalDist
        hd = HierarchicalDist(f, primary_dist)
        snt_hd = HierarchicalDist(f, snt_primary_dist)

        @test @inferred(unshaped(hd)) isa ValueShapes.UnshapedHDist
        ud = unshaped(hd)

        @test @inferred(rand(hd)) isa NamedTuple
        @test @inferred(rand(snt_hd)) isa ShapedAsNT
        @test @inferred(rand(ud)) isa AbstractVector{<:Real}
        @test @inferred(varshape(hd)) == NamedTupleShape(foo = ScalarShape{Real}(), bar = ScalarShape{Real}(), baz = ArrayShape{Real}(3))
        @test @inferred(varshape(ud)) == ArrayShape{Real}(5)

        ux = [2.7, 4.3, 8.7, 8.7, 8.7]
        @test @inferred(logpdf(ud, ux)) ≈ logpdf(primary_dist.foo, 2.7) + logpdf(primary_dist.bar, 4.3) + 3 * logpdf(Normal(4.3, 2.7), 8.7)
        @test @inferred(logpdf(hd, varshape(hd)(ux))) == logpdf(ud, ux)
        @test @inferred(logpdf(hd, varshape(hd)(ux))) == logpdf(ud, ux)

        stdnormal = StdNormal()^getdof(hd)
        #test_transport(stdnormal, hd)
        #test_transport(hd, stdnormal)
    end

    let
        hd = HierarchicalDist(
            v -> NamedTupleDist(b = Normal(v.a, 1.2)),
            NamedTupleDist(a = Normal(2.3, 1.9))
        )

        cov_expected = [1.9^2 1.9^2; 1.9^2 1.9^2 + 1.2^2]

        @test isapprox(cov(unshaped(hd)), cov_expected, rtol = 0.05)
        @test isapprox(mean(unshaped.(rand(hd, 10^5))), [2.3, 2.3], rtol = 0.05)
    end
end
