# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Statistics, Random
using StatsBase, Distributions, ArraysOfArrays, IntervalSets

using MeasureBase

@testset "vartransform" begin
    function test_back_and_forth(trg, src)
        @testset "transform $(typeof(trg).name) <-> $(typeof(src).name)" begin
            x = rand(src)
            y = vartransform_def(trg, src, x)
            src_v_reco = vartransform_def(src, trg, y)
    
            @test x ≈ src_v_reco
    
            let vs_trg = varshape(trg), vs_src = varshape(src)
                f = unshaped_x -> inverse(vs_trg)(vartransform_def(trg, src, vs_src(unshaped_x)))
                ref_ladj = logpdf(src, x) - logpdf(trg, y)
                @test ref_ladj ≈ logabsdet(ForwardDiff.jacobian(f, inverse(vs_src)(x)))[1]
            end
        end
    end
    
    ntdist = NamedTupleDist(
        a = uniform1,
        b = mvnorm,
        c = [4.2, 3.7],
        x = beta,
        y = gamma
    )

    stdmvnorm2 = StandardDist{Normal}(2)
    standnorm2_reshaped = ReshapedDist(stdmvnorm2, varshape(stdmvnorm2))

    test_back_and_forth(ntdist, StandardDist{Normal}(5))
    test_back_and_forth(ntdist, StandardDist{Uniform}(5))

    test_back_and_forth(stdmvnorm2, standnorm2_reshaped)

    
    for VT in (NamedTuple, ShapedAsNT)
        src_dist = unshaped(NamedTupleDist(VT, a = Weibull(), b = MvNormal([1.3 0.6; 0.6 2.4])))
        f = vartransform(Normal, src_dist)
        x = rand(src_dist)
        InverseFunctions.test_inverse(f, x)
        ChangesOfVariables.test_with_logabsdet_jacobian(f, x, ForwardDiff.jacobian)
    end
    
    
    @testset "trafo composition" begin
        dist1 = @inferred(NamedTupleDist(a = Normal(), b = Uniform(), c = Cauchy()))
        dist2 = @inferred(NamedTupleDist(a = Exponential(), b = Weibull(), c = Beta()))
        normal1 = Normal()
        normal2 = Normal(2)
    
        trafo = @inferred(vartransform(dist1, dist2))
        inv_trafo = @inferred(inverse(trafo))
    
        composed_trafo = @inferred(∘(trafo, inv_trafo))
        @test composed_trafo.source_dist == composed_trafo.target_dist == dist1
        @test composed_trafo ∘ trafo == trafo
        @test_throws ArgumentError  trafo ∘ composed_trafo
    
        trafo = @inferred(vartransform(normal1, normal2))
        @test_throws ArgumentError trafo ∘ trafo
    end
end
