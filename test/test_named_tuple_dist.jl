# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Statistics, Random
using StatsBase, Distributions, ArraysOfArrays, IntervalSets
import ForwardDiff

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

    let X = varshape(dist).(VectorOfSimilarVectors(Array{eltype(unshaped(dist))}(undef, length(unshaped(dist)), 14)))
        @test @inferred(rand!(dist, X[1])) == X[1]
        @test @inferred(rand!(dist, X)) === X
    end

    let X = varshape(dist).([Array{Float64}(undef, totalndof(varshape(dist))) for i in 1:11])
        @test @inferred(rand!(dist, X[1])) == X[1]
        @test @inferred(rand!(dist, X)) === X
    end

    testrng() = MersenneTwister(0xaef035069e01e678)

    let X = rand(unshaped(dist), 10), Xn = VectorOfSimilarVectors(X)
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
    @test shape.(VectorOfSimilarVectors(@inferred(rand(testrng(), unshaped(dist), 10^3)))) == @inferred(rand(testrng(), dist, 10^3))

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
end

@testset "empty NamedTupleDist" begin
    d = NamedTupleDist()
    @test totalndof(varshape(d)) == 0
    @test length(unshaped(d)) == 0
    @test rand(d) == NamedTuple()
    @test isempty(rand(unshaped(d)))
    @test logpdf(d, NamedTuple()) == 0
    @test insupport(d, NamedTuple()) === true
    @test isempty(mean(unshaped(d)))

    dn = NamedTupleDist(x = NamedTuple(), y = Normal())
    @test length(unshaped(dn)) == 1
    @test logpdf(dn, mean(dn)) ≈ logpdf(Normal(), 0.0)
    @test logpdf(unshaped(dn), [0.0]) ≈ logpdf(Normal(), 0.0)
    @test insupport(dn, (x = NamedTuple(), y = 0.5)) === true
    @test rand(dn).x == NamedTuple()
end

@testset "NamedTupleDist with nested named tuples" begin
    # Flat parameter vector: (a, c, e, f). Groups b = (c, d) and d = (e,) are
    # nested slices of it.
    dist = NamedTupleDist((a = 0..1, b = (c = 0..2, d = (e = 0..3,)), f = 0..1))
    shape = varshape(dist)
    x = (a = 0.5, b = (c = 1.0, d = (e = 2.0,)), f = 0.3)
    x_unshaped = [0.5, 1.0, 2.0, 0.3]

    @test dist == NamedTupleDist(a = 0..1, b = NamedTupleDist(c = 0..2, d = NamedTupleDist(e = 0..3)), f = 0..1)
    @test totalndof(shape) == 4
    @test @inferred(length(unshaped(dist))) == 4
    @test @inferred(unshaped(x, shape)) == x_unshaped
    @test @inferred(shape(x_unshaped)) == x

    # Densities are uniform: 1 / (1 * 2 * 3 * 1)
    @test @inferred(logpdf(dist, x)) ≈ -log(6)
    @test @inferred(logpdf(dist, shape(x_unshaped))) ≈ -log(6)
    @test @inferred(logpdf(dist, ShapedAsNT(x_unshaped, shape))) ≈ -log(6)
    @test @inferred(logpdf(unshaped(dist), x_unshaped)) ≈ -log(6)
    @test insupport(dist, x)
    @test !insupport(dist, (a = 0.5, b = (c = 3.0, d = (e = 2.0,)), f = 0.3))
    @test @inferred(insupport(unshaped(dist), x_unshaped))
    @test !insupport(unshaped(dist), [0.5, 3.0, 2.0, 0.3])

    @test @inferred(mean(unshaped(dist))) ≈ [0.5, 1.0, 1.5, 0.5]
    @test @inferred(mode(unshaped(dist))) ≈ [0.5, 1.0, 1.5, 0.5]
    @test @inferred(var(unshaped(dist))) ≈ [1/12, 4/12, 9/12, 1/12]
    @test mean(dist) == (a = 0.5, b = (c = 1.0, d = (e = 1.5,)), f = 0.5)
    @test mode(dist) == (a = 0.5, b = (c = 1.0, d = (e = 1.5,)), f = 0.5)
    @test var(dist) == (a = 1/12, b = (c = 4/12, d = (e = 9/12,)), f = 1/12)

    # Statistics keep the number types of the components:
    dist32 = NamedTupleDist(a = Normal(1f0, 2f0), b = (c = Normal(0f0, 1f0), k = 3))
    @test @inferred(mean(unshaped(dist32))) isa Vector{Float32}
    @test @inferred(var(unshaped(dist32))) isa Vector{Float32}
    @test @inferred(mode(unshaped(dist32))) isa Vector{Float32}
    @test mode(unshaped(NamedTupleDist(n = Poisson(3.0), x = (k = 3,)))) isa Vector{Int}
    @test mode(unshaped(NamedTupleDist(n = Poisson(3.0), x = (k = 3,)))) == [3]
    @test mean(unshaped(NamedTupleDist(k = 3, x = (l = [1, 2],)))) isa Vector{Bool}
    @test isempty(mean(unshaped(NamedTupleDist(k = 3, x = (l = [1, 2],)))))
    @test mean(NamedTupleDist(a = Normal(big"1e400", big"1"), b = (c = Normal(),))).a == big"1e400"
    @test ForwardDiff.derivative(t -> mean(NamedTupleDist(a = Normal(t, 1.0), b = (c = Normal(0.0, t),))).a, 2.0) == 1.0
    @test ForwardDiff.derivative(t -> var(NamedTupleDist(a = Normal(t, 1.0), b = (c = Normal(0.0, t),))).b.c, 2.0) == 4.0

    rng = MersenneTwister(123)
    r = @inferred rand(rng, dist)
    @test r isa NamedTuple && r.b isa NamedTuple && r.b.d isa NamedTuple
    @test insupport(dist, r)
    R = rand(rng, dist, 100)
    @test R.b isa ShapedAsNTArray
    @test eltype(R) == ValueShapes.shaped_type(shape)
    @test collect(R) isa Vector{eltype(R)}
    @test all(r -> insupport(dist, r), R)
    @test all(0 .<= R.b.d.e .<= 3)
    R2 = rand(rng, dist, (3, 2))
    @test R2.b isa ShapedAsNTArray{<:NamedTuple,2}
    @test size(R2.b.d.e) == (3, 2)

    # Variates with non-1-based indices:
    buf = view(fill(NaN, 6), Base.IdentityUnitRange(2:5))
    rand!(rng, unshaped(dist), buf)
    @test insupport(unshaped(dist), buf)
    @test logpdf(unshaped(dist), buf) ≈ -log(6)

    # The value type is per nesting level:
    sdist = NamedTupleDist(ShapedAsNT, a = 0..1, b = (c = 0..2,), d = NamedTupleDist(ShapedAsNT, e = 0..3))
    sr = rand(rng, sdist)
    @test sr isa ShapedAsNT
    @test sr.b isa NamedTuple
    @test sr.d isa ShapedAsNT
    @test sr[].b isa NamedTuple
    @test @inferred(logpdf(sdist, sr)) ≈ -log(6)
    @test @inferred(insupport(sdist, sr))

    # Nested components with constants and correlated multivariate distributions:
    cdist = NamedTupleDist(x = NamedTupleDist(a = Normal(1, 2), b = MvNormal([2.0 0.5; 0.5 1.0])), k = 7, c = Normal(0, 3))
    @test @inferred(length(unshaped(cdist))) == 4
    @test @inferred(cov(unshaped(cdist))) ≈ [4.0 0.0 0.0 0.0; 0.0 2.0 0.5 0.0; 0.0 0.5 1.0 0.0; 0.0 0.0 0.0 9.0]
    @test @inferred(var(unshaped(cdist))) ≈ [4.0, 2.0, 1.0, 9.0]
    @test mean(cdist) == (x = (a = 1.0, b = [0.0, 0.0]), k = 7, c = 0.0)
    @test var(cdist) == (x = (a = 4.0, b = [2.0, 1.0]), k = 0, c = 9.0)
    @test mode(cdist) == (x = (a = 1.0, b = [0.0, 0.0]), k = 7, c = 0.0)
    cv = rand(rng, unshaped(cdist))
    @test @inferred(logpdf(unshaped(cdist), cv)) ≈ logpdf(Normal(1, 2), cv[1]) + logpdf(MvNormal([2.0 0.5; 0.5 1.0]), cv[2:3]) + logpdf(Normal(0, 3), cv[4])
    @test @inferred(insupport(unshaped(cdist), cv))
    @test logpdf(cdist, varshape(cdist)(cv)) == logpdf(unshaped(cdist), cv)
    @test logpdf(cdist, (x = (a = cv[1], b = cv[2:3]), k = 7, c = cv[4])) == logpdf(unshaped(cdist), cv)

    # Same number of components on both levels, so the nested reductions re-enter
    # the same methods with a view of the flat vector:
    ndist = NamedTupleDist(a = Normal(), b = NamedTupleDist(c = Normal(1, 2), g = Uniform(0, 1)))
    nv = rand(rng, unshaped(ndist))
    @test @inferred(logpdf(unshaped(ndist), nv)) ≈ logpdf(Normal(), nv[1]) + logpdf(Normal(1, 2), nv[2]) + logpdf(Uniform(0, 1), nv[3])
    @test @inferred(insupport(unshaped(ndist), nv))
    @test @inferred(logpdf(ndist, varshape(ndist)(nv))) == logpdf(unshaped(ndist), nv)
    @test @inferred(insupport(ndist, varshape(ndist)(nv)))
end
