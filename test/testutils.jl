# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

import MeasureBase
import ForwardDiff

function test_transport(ν, μ)
    supertype(x::Real) = Real
    supertype(x::AbstractArray{<:Real,N}) where N = AbstractArray{<:Real,N}

    @testset "transport from $(nameof(typeof(ν))) to $(nameof(typeof(μ)))" begin
        x = rand(μ)
        @test !(@inferred(MeasureBase.transport_to(ν, μ)(x)) isa NoVarTransform)
        f = MeasureBase.transport_to(ν, μ)
        y = f(x)
        @test @inferred(inverse(f)(y)) ≈ x
        @test @inferred(with_logabsdet_jacobian(f, x)) isa Tuple{supertype(y),Real}
        @test @inferred(with_logabsdet_jacobian(inverse(f), y)) isa Tuple{supertype(x),Real}
        y2, ladj_fwd = with_logabsdet_jacobian(f, x)
        x2, ladj_inv = with_logabsdet_jacobian(inverse(f), y)
        
        @test x ≈ x2
        @test y ≈ y2
        @test ladj_fwd ≈ - ladj_inv
        @test ladj_fwd ≈ logdensityof(μ, x) - logdensityof(ν, y)

        vs_μ = varshape(μ)
        vs_ν = varshape(ν)
        ladj_fwd_ad = logabsdet(ForwardDiff.jacobian(inverse(vs_ν) ∘ f ∘ vs_μ, inverse(vs_μ)(x)))
        ladj_inv_ad = logabsdet(ForwardDiff.jacobian(inverse(vs_μ) ∘ inverse(f) ∘ vs_ν, inverse(vs_ν)(y)))
        @test ladj_fwd ≈ ladj_fwd_ad
        @test ladj_inv ≈ ladj_inv_ad
    end
end
