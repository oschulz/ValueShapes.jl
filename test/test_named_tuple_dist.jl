# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Statistics, Random
using StatsBase, Distributions, ArraysOfArrays, IntervalSets

@testset "NamedTupleDist" begin
    dist = @inferred NamedTupleDist(
        ShapedAsNT,
        a = 5, b = Weibull(2, 1),
        c = -4..5,
        d = MvNormal([1.2 0.5; 0.5 2.1]),
        x = [3 4; 2 5],
        e = [Normal(1.1, 0.2)]
    )

    @test typeof(@inferred varshape(dist)) <: NamedTupleShape

    shape = varshape(dist)
    shapes = map(varshape, dist)

    for k in keys(dist)
        @test getproperty(shapes, k) == varshape(getproperty(dist, k))
    end

    @test dist[:d] == dist.d

    X_unshaped = [0.2, -0.4, 0.3, -0.5, 0.9]
    X_shaped = shape(X_unshaped)
    @test (@inferred logpdf(unshaped(dist), X_unshaped)) == logpdf(Weibull(2, 1), 0.2) + logpdf(Uniform(-4, 5), -0.4) + logpdf(MvNormal([1.2 0.5; 0.5 2.1]), [0.3, -0.5]) + logpdf(Normal(1.1, 0.2), 0.9)
    @test (@inferred logpdf(dist, X_shaped)) == logpdf(unshaped(dist), X_unshaped)
    @test (@inferred logpdf(dist, X_shaped[])) == logpdf(unshaped(dist), X_unshaped)

    @test (@inferred mode(unshaped(dist))) == [mode(dist.b), 0.5, 0.0, 0.0, 1.1]
    @test (@inferred mode(dist)) == shape(mode(unshaped(dist)))

    @test (@inferred mean(unshaped(dist))) == [mean(dist.b), 0.5, 0.0, 0.0, 1.1]
    @test (@inferred mean(dist)) == shape(mean(unshaped(dist)))

    @test @inferred(var(unshaped(dist))) ≈ [var(dist.b), 6.75, 1.2, 2.1, 0.04]
    @test @inferred(var(dist))[] == (a = 0, b = var(dist.b), c = var(dist.c), d = var(dist.d), x = var(dist.x), e = var(dist.e))

    @test begin
        ref_cov =
            [var(dist.b)  0.0   0.0  0.0 0.0;
             0.0  6.75  0.0  0.0 0.0;
             0.0  0.0   1.2  0.5 0.0;
             0.0  0.0   0.5  2.1 0.0;
             0.0  0.0   0.0  0.0 0.04 ]

        (@inferred cov(unshaped(dist))) ≈ ref_cov
    end

    @test @inferred(rand(dist)) isa ShapedAsNT
    @test @inferred(rand(dist)[]) isa NamedTuple
    @test pdf(dist, rand(dist)) > 0
    @test @inferred(rand(dist, ())) isa ShapedAsNTArray{T,0} where T
    @test pdf(dist, rand(dist)) > 0
    @test @inferred(rand(dist, 100)) isa ShapedAsNTArray
    @test all(x -> x > 0, pdf.(Ref(dist), rand(dist, 10^3)))

    let X = varshape(dist).(nestedview(Array{eltype(unshaped(dist))}(undef, length(unshaped(dist)), 14)))
        @test @inferred(rand!(dist, X[1])) == X[1]
        @test @inferred(rand!(dist, X)) === X
    end

    let X = varshape(dist).([Array{Float64}(undef, totalndof(varshape(dist))) for i in 1:11])
        @test @inferred(rand!(dist, X[1])) == X[1]
        @test @inferred(rand!(dist, X)) === X
    end

    testrng() = MersenneTwister(0xaef035069e01e678)

    let X = rand(unshaped(dist), 10), Xn = nestedview(X)
        @test @inferred(Distributions._pdf(unshaped(dist), Xn[1])) == @inferred(pdf(unshaped(dist), Xn[1]))
        @test @inferred(pdf(unshaped(dist), X)) == @inferred(broadcast(pdf, Ref(unshaped(dist)), Xn))

        @test @inferred(Distributions._logpdf(unshaped(dist), Xn[1])) == @inferred(logpdf(unshaped(dist), Xn[1]))
        @test @inferred(logpdf(unshaped(dist), X)) == @inferred(broadcast(logpdf, Ref(unshaped(dist)), Xn))

        @test @inferred(insupport(unshaped(dist), Xn[1])) == true
        @test @inferred(insupport(unshaped(dist), fill(-Inf, length(Xn)))) == false
        @test @inferred(insupport(unshaped(dist), X)) == fill(true, length(Xn))
    end

    @test @inferred(rand(unshaped(dist))) isa Vector{Float64}
    @test shape(@inferred(rand(testrng(), unshaped(dist)))) == @inferred(rand(testrng(), dist, ()))[] == @inferred(rand(testrng(), dist))
    @test @inferred(rand(unshaped(dist), 10^3)) isa Matrix{Float64}
    @test shape.(nestedview(@inferred(rand(testrng(), unshaped(dist), 10^3)))) == @inferred(rand(testrng(), dist, 10^3))

    propnames = propertynames(dist, true)
    @test propnames == (:a, :b, :c, :d, :x, :e, :_internal_distributions, :_internal_shape)
    @test @inferred(keys(dist)) == propertynames(dist, false)
    internaldists = getproperty(dist, :_internal_distributions)
    internalshape = getproperty(dist, :_internal_shape)
    @test all(i -> typeof(getindex(internaldists, i)) == typeof(getproperty(dist, keys(dist)[i])), 1:length(internaldists))
    @test all(key -> getproperty(getproperty(internalshape, key), :shape) == valshape(getproperty(internalshape, key)), keys(internalshape))

    @test @inferred(convert(NamedTupleDist, (x = 5, z = Normal()))) == NamedTupleDist(x = 5, z = Normal())
    @test @inferred(merge((a = 42,), NamedTupleDist(x = 5, z = Normal()))) == (a = 42, x = ConstValueDist(5), z = Normal())
    @test @inferred(NamedTupleDist(ShapedAsNT, (;dist...))) == dist
    @test @inferred(merge(dist)) === dist
    @test @inferred(merge(
        NamedTupleDist(x = Normal(), y = 42),
        NamedTupleDist(y = Normal(4, 5)),
        NamedTupleDist(z = Exponential(3), a = 4.2),
    )) == NamedTupleDist(x = Normal(), y = Normal(4, 5), z = Exponential(3), a = 4.2)
    @test @inferred(merge(NamedTupleDist(a = 42,), (x = 5, z = Normal()))) == NamedTupleDist(a = 42, x = ConstValueDist(5), z = Normal())
    @test @inferred(merge(
        NamedTupleDist(x = Normal(), y = 42),
        (y = Normal(4, 5),),
        NamedTupleDist(z = Exponential(3), a = 4.2),
    )) == NamedTupleDist(x = Normal(), y = Normal(4, 5), z = Exponential(3), a = 4.2)

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
        
        test_back_and_forth(ntdist, StandardDist{Normal}(5))
        test_back_and_forth(ntdist, StandardDist{Uniform}(5))
        
        
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
end
