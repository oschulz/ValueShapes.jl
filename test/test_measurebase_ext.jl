# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions
using LinearAlgebra
using MeasureBase
using MeasureBase: StdNormal, StdUniform, Dirac, asmeasure, productmeasure, testvalue, pushfwd, transport_to

@testset "MeasureBase extension" begin
    @testset "varshape" begin
        @test varshape(asmeasure(Normal())) == ScalarShape{Real}()
        @test varshape(asmeasure(MvNormal(zeros(2), Matrix(I, 2, 2)))) == ArrayShape{Real}(2)
        @test varshape(StdNormal()^3) == ArrayShape{Real}(3)
        @test varshape(StdNormal()^(2, 3)) == ArrayShape{Real}(2, 3)
        @test varshape(productmeasure([asmeasure(Normal(μ, 1.0)) for μ in 1:3])) == ArrayShape{Real}(3)
        @test varshape(Dirac(4.2)) == ConstValueShape(4.2)
        @test varshape(weightedmeasure(0.3, StdUniform()^2)) == ArrayShape{Real}(2)
        μ = productmeasure((a = asmeasure(Normal()), b = StdUniform()^2, c = Dirac(1.5)))
        @test varshape(μ) == NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Real}(2), c = ConstValueShape(1.5))
        @test totalndof(varshape(μ)) == 3
        @test varshape(mintegrate_exp(x -> 0.0, μ)) == varshape(μ)
    end

    @testset "unshaped measures" begin
        μ = productmeasure((a = asmeasure(Normal(0.5, 2.0)), b = StdUniform()^2, c = Dirac(1.5)))
        x = (a = 0.4, b = [0.2, 0.7], c = 1.5)
        xf = unshaped(x, varshape(μ))
        @test xf == [0.4, 0.2, 0.7]
        @test logdensityof(unshaped(μ), xf) ≈ logdensityof(μ, x)
        @test unshaped(StdNormal()^3) === StdNormal()^3
        @test logdensityof(unshaped(asmeasure(Normal(0.5, 2.0))), [0.4]) ≈ logpdf(Normal(0.5, 2.0), 0.4)
        @test varshape(unshaped(μ)) == ArrayShape{Real}(3)
        @test logdensityof(varshape(μ)(unshaped(μ)), x) ≈ logdensityof(μ, x)
        @test unshaped(μ, varshape(μ)) == unshaped(μ)
        @test_throws ArgumentError unshaped(μ, ArrayShape{Real}(2))
        @test resultshape(transport_to(μ, StdNormal()^3), ArrayShape{Real}(3)) == varshape(μ)
        @test varshape(pushfwd(x -> 2 .* x, StdNormal()^2)) == ArrayShape{Real}(2)
    end

    @testset "distributions as measures" begin
        ntd = NamedTupleDist(a = Normal(0.3, 1.2), b = MvNormal([0.1, 0.2], [1.0 0.2; 0.2 1.0]), c = 2.5)
        m = asmeasure(ntd)
        @test m isa MeasureBase.ProductMeasure{<:NamedTuple}
        @test varshape(m) == varshape(ntd)
        x = (a = 0.4, b = [0.5, -0.2], c = 2.5)
        @test logdensityof(m, x) ≈ logpdf(ntd, x)
        @test logdensityof(m, (a = 0.4, b = [0.5, -0.2], c = 2.0)) == -Inf
        rd = ReshapedDist(MvNormal(zeros(4), Matrix(I, 4, 4)), ArrayShape{Real}(2, 2))
        X = [0.1 0.2; 0.3 0.4]
        @test varshape(asmeasure(rd)) == ArrayShape{Real}(2, 2)
        @test logdensityof(asmeasure(rd), X) ≈ logpdf(rd, X)
        @test asmeasure(ConstValueDist(3.0)) == Dirac(3.0)
    end
end
