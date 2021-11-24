# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    ConstValueDist <: Distributions.Distribution

Represents a delta distribution for a constant value of arbritrary type.

Calling `varshape` on a `ConstValueDist` will yield a
[`ConstValueShape`](@ref).
"""
struct ConstValueDist{VF<:VariateForm,T} <: Distribution{VF,Discrete}
    value::T
end

export ConstValueDist

ConstValueDist(x::T) where {T<:Real} = ConstValueDist{Univariate,T}(x)
ConstValueDist(x::T) where {T<:AbstractVector{<:Real}} = ConstValueDist{Multivariate,T}(x)
ConstValueDist(x::T) where {T<:AbstractMatrix{<:Real}} = ConstValueDist{Matrixvariate,T}(x)
ConstValueDist(x::NamedTuple{names}) where names = ConstValueDist{NamedTupleVariate{names},typeof(x)}(x)


_pdf_impl(d::ConstValueDist, x) = d.value == x ? float(eltype(d))(1) : float(eltype(d))(0)
_logpdf_impl(d::ConstValueDist, x) = d.value == x ? float(eltype(d))(0) : float(eltype(d))(-Inf)

Distributions.pdf(d::ConstValueDist{Univariate}, x::Real) = _pdf_impl(d, x)
Distributions.logpdf(d::ConstValueDist{Univariate}, x::Real) = _logpdf_impl(d, x)

Distributions._pdf(d::ConstValueDist{Multivariate}, x::AbstractVector{<:Real}) = _pdf_impl(d, x)
Distributions._logpdf(d::ConstValueDist{Multivariate}, x::AbstractVector{<:Real}) = log(pdf(d, x))

Distributions._pdf(d::ConstValueDist{Matrixvariate}, x::AbstractMatrix{<:Real}) = _pdf_impl(d, x)
Distributions._logpdf(d::ConstValueDist{Matrixvariate}, x::AbstractMatrix{<:Real}) = log(pdf(d, x))

Distributions.pdf(d::ConstValueDist{<:NamedTupleVariate{names}}, x::NamedTuple{names}) where names = _pdf_impl(d, x)
Distributions.logpdf(d::ConstValueDist{<:NamedTupleVariate{names}}, x::NamedTuple{names}) where names = log(pdf(d, x))


Distributions.insupport(d::ConstValueDist{Univariate}, x::Real) = x == d.value
Distributions.insupport(d::ConstValueDist{Multivariate}, x::AbstractVector{<:Real}) = x == d.value
Distributions.insupport(d::ConstValueDist{Matrixvariate}, x::AbstractMatrix{<:Real}) = x == d.value
Distributions.insupport(d::ConstValueDist{<:NamedTupleVariate{names}}, x::NamedTuple{names}) where names = x == d.value

Distributions.cdf(d::ConstValueDist{Univariate}, x::Real) = d.value <= x ? Float32(1) : Float32(0)
Distributions.quantile(d::ConstValueDist{Univariate}, q::Real) = d.value # Sensible?
Distributions.minimum(d::ConstValueDist{Univariate}) = d.value
Distributions.maximum(d::ConstValueDist{Univariate}) = d.value

StatsBase.mean(d::ConstValueDist) = d.value

StatsBase.mode(d::ConstValueDist) = d.value

Base.size(d::ConstValueDist{<:PlainVariate}) = size(d.value)
Base.length(d::ConstValueDist{<:PlainVariate}) = prod(size(d))
Base.eltype(d::ConstValueDist{<:PlainVariate}) = eltype(d.value)


Random.rand(rng::AbstractRNG, d::ConstValueDist) = d.value

Distributions._rand!(rng::AbstractRNG, d::ConstValueDist{<:Multivariate}, x::AbstractVector{<:Real}) = copyto!(x, d.value)
Distributions._rand!(rng::AbstractRNG, d::ConstValueDist{<:Matrixvariate}, x::AbstractMatrix{<:Real}) = copyto!(x, d.value)

Random.rand(rng::AbstractRNG, d::ConstValueDist{<:StructVariate}, dims::Dims) = Fill(d.value, dims)
Random.rand!(rng::AbstractRNG, d::ConstValueDist{<:StructVariate}, A::AbstractArray) = fill!(A, d.value)


ValueShapes.varshape(d::ConstValueDist) = ConstValueShape(d.value)

Statistics.var(d::ConstValueDist) = zero(d.value)
