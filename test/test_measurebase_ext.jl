# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Distributions
using LinearAlgebra
using Random: Xoshiro
using InverseFunctions: inverse
using Static: static
using MeasureBase
using MeasureBase: StdNormal, StdUniform, Dirac, asmeasure, productmeasure, testvalue, pushfwd, transport_to, mbind

@testset "MeasureBase extension" begin
    @testset "varshape" begin
        @test varshape(asmeasure(Normal())) == ScalarShape{Real}()
        @test varshape(asmeasure(MvNormal(zeros(2), Matrix(I, 2, 2)))) == ArrayShape{Real}(2)
        @test varshape(StdNormal()^3) == ArrayShape{Real}(3)
        @test varshape(StdNormal()^(2, 3)) == ArrayShape{Real}(2, 3)
        @test varshape(StdNormal()^static(3)) == ArrayShape{Real}(3)
        @test varshape(StdNormal()^(static(2), static(3))) == ArrayShape{Real}(2, 3)
        @test varshape(productmeasure([asmeasure(Normal(μ, 1.0)) for μ in 1:3])) == ArrayShape{Real}(3)
        @test varshape(Dirac(4.2)) == ConstValueShape(4.2)
        @test varshape(weightedmeasure(0.3, StdUniform()^2)) == ArrayShape{Real}(2)
        μ = productmeasure((a = asmeasure(Normal()), b = StdUniform()^2, c = Dirac(1.5)))
        @test varshape(μ) == NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Real}(2), c = ConstValueShape(1.5))
        @test totalndof(varshape(μ)) == 3
        @test varshape(mintegrate_exp(x -> 0.0, μ)) == varshape(μ)
    end

    @testset "binds" begin
        U = StdUniform()
        f_β(a) = productmeasure((b = U^2, d = Dirac(2.0)))
        μ = mbind(f_β, productmeasure((a = U, c = Dirac(1.5))), merge)
        vs_μ = NamedTupleShape(a = ScalarShape{Real}(), c = ConstValueShape(1.5), b = ArrayShape{Real}(2), d = ConstValueShape(2.0))
        @test varshape(μ) == vs_μ
        @test totalndof(varshape(μ)) == 3
        # Nested binds, as primary and as secondary measure, and binds in products:
        vs_e = NamedTupleShape(e = ScalarShape{Real}(), g = ConstValueShape(0.1))
        @test varshape(mbind(a -> productmeasure((e = U, g = Dirac(0.1))), μ, merge)) == merge(vs_μ, vs_e)
        @test varshape(mbind(a -> μ, productmeasure((e = U, g = Dirac(0.1))), merge)) == merge(vs_e, vs_μ)
        @test varshape(productmeasure((p = U, q = μ))) == NamedTupleShape(p = ScalarShape{Real}(), q = vs_μ)
        @test varshape(mbind(a -> U^2, U^1, vcat)) == ArrayShape{Real}(3)
        @test varshape(mbind(a -> U^2, U, vcat)) == ArrayShape{Real}(3)
        @test varshape(mbind(f_β, U)) == varshape(f_β(0.5))
        # Binds with value-dependent secondary shapes have no fixed shape:
        counts = asmeasure(Poisson(1000.0))
        @test_throws ArgumentError varshape(mbind(n -> StdNormal()^n, counts))
        @test_throws ArgumentError varshape(mbind(n -> StdNormal()^n, counts, vcat))
        @test_throws ArgumentError varshape(mbind(a -> productmeasure((b = Dirac(a.a),)), productmeasure((a = U,)), merge))
        @test_throws ArgumentError varshape(mbind(a -> U, U, tuple))
    end

    @testset "flat layout" begin
        U = StdUniform()
        bnd = mbind(a -> productmeasure((b = U^2, d = Dirac(2.0))), productmeasure((a = U, c = Dirac(1.5))), merge)
        hand_built = [
            productmeasure((a = U, c = Dirac(1.5), b = U^(2, 2))) =>
                ((a = 0.1, c = 1.5, b = [0.2 0.4; 0.3 0.5]), [0.1, 0.2, 0.3, 0.4, 0.5]),
            productmeasure((a = productmeasure((x = U, y = Dirac(2.0), z = U^2)), b = U)) =>
                ((a = (x = 0.1, y = 2.0, z = [0.2, 0.3]), b = 0.4), [0.1, 0.2, 0.3, 0.4]),
            bnd => ((a = 0.1, c = 1.5, b = [0.2, 0.3], d = 2.0), [0.1, 0.2, 0.3]),
            productmeasure((p = U, q = bnd)) =>
                ((p = 0.7, q = (a = 0.1, c = 1.5, b = [0.2, 0.3], d = 2.0)), [0.7, 0.1, 0.2, 0.3]),
        ]
        @testset "$(nameof(typeof(μ)))" for (μ, (x, x_flat)) in hand_built
            @test unshaped(x, varshape(μ)) == x_flat
        end

        # Transports between standard uniforms are identities, so for measures
        # built from them, the transport to a standard uniform exposes
        # MeasureBase's flat layout:
        measures = [
            productmeasure((a = U, b = U)), U^3, U^(2, 3), U^static(3), U^(static(2), static(2)),
            first.(hand_built)..., mbind(a -> productmeasure((e = U^2,)), bnd, merge), mbind(a -> U^2, U^1, vcat),
        ]
        @testset "$(nameof(typeof(μ)))" for μ in measures
            vs = varshape(μ)
            x = rand(Xoshiro(1), μ)
            x_flat = unshaped(x, vs)
            @test getdof(μ) isa MeasureBase.NoDOF || getdof(μ) == totalndof(vs)
            f = transport_to(U^totalndof(vs), μ)
            @test f(x) ≈ x_flat
            @test unshaped(inverse(f)(x_flat), vs) ≈ x_flat
            @test vs(x_flat) == x
        end
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
