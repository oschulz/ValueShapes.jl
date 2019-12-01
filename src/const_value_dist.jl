# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    ConstValueDist <: Distributions.Distribution

Represents a delta distribution for a constant value of arbritrary type.

Calling `varshape` on a `ConstValueDist` will yield a
[`ConstValueShape`](@ref).
"""
struct ConstValueDist{VF<:VariateForm,T} <: Distribution{VF,Continuous}
    value::T
end

export ConstValueDist

ConstValueDist(x::T) where {T<:Real} = ConstValueDist{Univariate,T}(x)
ConstValueDist(x::T) where {T<:AbstractVector{<:Real}} = ConstValueDist{Multivariate,T}(x)
ConstValueDist(x::T) where {T<:AbstractMatrix{<:Real}} = ConstValueDist{Matrixvariate,T}(x)

Distributions.pdf(d::ConstValueDist{Univariate}, x::Real) = d.value == x ? float(eltype(d))(Inf) : float(eltype(d))(0)
Distributions._logpdf(d::ConstValueDist, x::AbstractArray{<:Real}) = d.value == x ? float(eltype(d))(Inf) : float(eltype(d))(0)

Distributions.cdf(d::ConstValueDist{Univariate}, x::Real) = d.value <= x ? Float32(1) : Float32(0)
Distributions.quantile(d::ConstValueDist{Univariate}, q::Real) = d.value # Sensible?
Distributions.minimum(d::ConstValueDist{Univariate}) = x.value
Distributions.maximum(d::ConstValueDist{Univariate}) = x.value
Distributions.insupport(d::ConstValueDist{Univariate}, x::Real) = x == d.value

StatsBase.mode(d::ConstValueDist) = d.value

Base.size(d::ConstValueDist) = size(d.value)
Base.length(d::ConstValueDist) = prod(size(d))
Base.eltype(d::ConstValueDist) = eltype(d.value)

Random.rand(rng::AbstractRNG, d::ConstValueDist) = d.value

function Random._rand!(rng::AbstractRNG, d::ConstValueDist, x::AbstractArray{<:Real})
    copyto!(x, d.value)
end


ValueShapes.varshape(d::ConstValueDist) = ConstValueShape(d.value)
