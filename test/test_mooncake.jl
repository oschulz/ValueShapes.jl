# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using ArraysOfArrays, Distributions, IntervalSets, Tables
using LinearAlgebra: norm
import Mooncake, ForwardDiff


_mooncake_gradient(f, x::AbstractVector{<:Real}) =
    Mooncake.value_and_gradient!!(Mooncake.prepare_gradient_cache(f, x), f, x)[2][2]

@testset "mooncake" begin
    @test Base.get_extension(ValueShapes, :ValueShapesMooncakeExt) isa Module

    # The rule for _adignore_call must make AD ignore the wrapped code:
    f_adignore(x) = x[1] * ValueShapes._adignore_call(() -> x[2])
    @test @inferred(f_adignore([3.0, 4.0])) == 12.0
    @test _mooncake_gradient(f_adignore, [3.0, 4.0]) == [4.0, 0.0]

    vs = NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}(), b = ArrayShape{Real}(2, 3), c = ConstValueShape(4.2))
    n = totalndof(vs)

    dist = NamedTupleDist(ShapedAsNT, a = 5, b = Weibull(2, 1), c = -4..5, d = MvNormal([1.2 0.5; 0.5 2.1]), e = [Normal(1.1, 0.2)])
    ud = unshaped(dist)

    nvs = NamedTupleShape(a = ScalarShape{Real}(), x = NamedTupleShape(b = ArrayShape{Real}(2), k = ConstValueShape(4.2), c = ScalarShape{Real}()))
    nn = totalndof(nvs)
    ndist = NamedTupleDist(a = Normal(), x = (b = MvNormal([1.2 0.5; 0.5 2.1]), k = 4.2, c = Exponential()))
    nud = unshaped(ndist)

    testcases = [
        (x -> vs(x).a + sum(vs(x).b) + vs(x).c, rand(n)),
        (x -> norm(unshaped(vs(x))), rand(n)),
        (x -> norm(unshaped(vs(x)[], vs)), rand(n)),
        (x -> logpdf(ud, x), rand(length(ud))),
        (X_flat -> begin
            Y = vs.(VectorOfSimilarVectors(reshape(X_flat, n, 5)))
            cols = Tables.columns(Y)
            sum(sum.(cols.b)) + sum(cols.a)
        end, rand(n * 5)),
        (X_flat -> begin
            Y = vs.(VectorOfSimilarVectors(reshape(X_flat, n, 5)))
            sum(sum.(unshaped.(Y, Ref(vs))))
        end, rand(n * 5)),
        # Nested named tuples:
        (x -> nvs(x).a + sum(nvs(x).x.b) * nvs(x).x.c + nvs(x).x.k, rand(nn)),
        (x -> logpdf(nud, x), rand(length(nud))),
        (X_flat -> begin
            Y = nvs.(VectorOfSimilarVectors(reshape(X_flat, nn, 3)))
            cols = Tables.columns(Y)
            sum(cols.a) + sum(cols.x.c) + sum(sum.(cols.x.b))
        end, rand(nn * 3)),
    ]

    for (f, x) in testcases
        @test _mooncake_gradient(f, x) ≈ ForwardDiff.gradient(f, x)
    end
end
