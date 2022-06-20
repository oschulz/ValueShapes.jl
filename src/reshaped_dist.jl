# This file is a part of jl, licensed under the MIT License (MIT).

@static if hasmethod(reshape, Tuple{Distribution{Multivariate,Continuous}, Int, Int})
    _reshape_arraylike_dist(d::Distribution, sz::Integer...) = reshape(d, sz)
else
    _reshape_arraylike_dist(d::Distribution, sz1::Integer, sz2::Integer) = MatrixReshaped(d, sz1, sz2)
end


"""
    ReshapedDist <: Distribution

An multivariate distribution reshaped using a given
[`AbstractValueShape`](@ref).

Constructors:

```julia
    ReshapedDist(dist::MultivariateDistribution, shape::AbstractValueShape)
```

In addition, `MultivariateDistribution`s can be reshaped via

```julia
(shape::AbstractValueShape)(dist::MultivariateDistribution)
```

with the difference that

```julia
(shape::ArrayShape{T,1})(dist::MultivariateDistribution)
```

will return the original `dist` instead of a `ReshapedDist`.
"""
struct ReshapedDist{
    VF <: VariateForm,
    VS <: ValueSupport,
    D <: Distribution{Multivariate,VS},
    S <: AbstractValueShape
} <: Distribution{VF,VS}
    dist::D
    shape::S
end 

export ReshapedDist


_variate_form(shape::ScalarShape) = Univariate

@static if isdefined(Distributions, :ArrayLikeVariate)
    _variate_form(shape::ArrayShape{T,N}) where {T,N} = ArrayLikeVariate{N}
else
    _variate_form(shape::ArrayShape{T,1}) where T = Multivariate
    _variate_form(shape::ArrayShape{T,2}) where T = Matrixvariate
end

_variate_form(shape::NamedTupleShape{names}) where names = NamedTupleVariate{names}

_with_zeroconst(shape::AbstractValueShape) = replace_const_shapes(const_zero_shape, shape)


function ReshapedDist(dist::MultivariateDistribution{VS}, shape::AbstractValueShape) where {VS}
    @argcheck totalndof(varshape(dist)) == totalndof(shape)
    VF = _variate_form(shape)
    D = typeof(dist)
    S = typeof(shape)
    ReshapedDist{VF,VS,D,S}(dist, shape)
end


function (shape::ArrayShape{<:Real,1})(dist::MultivariateDistribution) where {T<:Real}
    @argcheck totalndof(varshape(dist)) == totalndof(shape)
    dist
end

(shape::ArrayShape{<:Real})(dist::MultivariateDistribution) = _reshape_arraylike_dist(dist, size(shape)...)

# ToDo: Enable when `reshape(::MultivariateDistribution, ())` becomes fully functional in Distributions:
#(shape::ScalarShape{<:Real})(dist::MultivariateDistribution) = _reshape_arraylike_dist(dist, size(shape)...)

(shape::AbstractValueShape)(dist::MultivariateDistribution) = ReshapedDist(dist, shape)


@inline varshape(rd::ReshapedDist) = rd.shape

@inline unshaped(rd::ReshapedDist) = rd.dist


Random.rand(rng::AbstractRNG, rd::ReshapedDist{Univariate}) = varshape(rd)(rand(rng, unshaped(rd)))

function Distributions._rand!(rng::AbstractRNG, rd::ReshapedDist{Multivariate}, x::AbstractVector{<:Real})
    Distributions._rand!(rng, unshaped(rd), x)
end

@static if isdefined(Distributions, :ArrayLikeVariate)
    function Distributions._rand!(rng::AbstractRNG, rd::ReshapedDist{<:ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where N
        Distributions._rand!(rng, _reshape_arraylike_dist(unshaped(rd), size(rd)...), x)
    end
else
    function Distributions._rand!(rng::AbstractRNG, rd::ReshapedDist{Matrixvariate}, x::AbstractMatrix{<:Real})
        Distributions._rand!(rng, _reshape_arraylike_dist(unshaped(rd), size(rd)...), x)
    end
end

Base.size(rd::ReshapedDist{<:PlainVariate}) = size(varshape(rd))

Base.length(rd::ReshapedDist{<:PlainVariate}) = prod(size(rd))

Statistics.mean(rd::ReshapedDist) = varshape(rd)(mean(unshaped(rd)))

StatsBase.mode(rd::ReshapedDist) = varshape(rd)(mode(unshaped(rd)))

Statistics.var(rd::ReshapedDist) = _with_zeroconst(varshape(rd))(var(unshaped(rd)))

Statistics.cov(rd::ReshapedDist{Multivariate}) = cov(unshaped(rd))


Distributions.pdf(rd::ReshapedDist{Univariate}, x::Real) = pdf(unshaped(rd), unshaped(x))
Distributions._pdf(rd::ReshapedDist{Multivariate}, x::AbstractVector{<:Real}) = pdf(unshaped(rd), x)
Distributions._pdf(rd::ReshapedDist{Matrixvariate}, x::AbstractMatrix{<:Real}) = pdf(_reshape_arraylike_dist(unshaped(rd), size(rd)...), x)

Distributions.logpdf(rd::ReshapedDist{Univariate}, x::Real) = logpdf(unshaped(rd), unshaped(x))
Distributions._logpdf(rd::ReshapedDist{Multivariate}, x::AbstractVector{<:Real}) = logpdf(unshaped(rd), x)
Distributions._logpdf(rd::ReshapedDist{Matrixvariate}, x::AbstractMatrix{<:Real}) = logpdf(_reshape_arraylike_dist(unshaped(rd), size(rd)...), x)

Distributions.insupport(rd::ReshapedDist{Univariate}, x::Real) = insupport(unshaped(rd), unshaped(x))
Distributions.insupport(rd::ReshapedDist{Multivariate}, x::AbstractVector{<:Real}) = insupport(unshaped(rd), x)
Distributions.insupport(rd::ReshapedDist{Matrixvariate}, x::AbstractMatrix{<:Real}) = insupport(_reshape_arraylike_dist(unshaped(rd), size(rd)...), x)


MeasureBase.getdof(μ::ReshapedDist) = getdof(unshaped(μ))

# Bypass `checked_var`, would require potentially costly transformation:
@inline MeasureBase.checked_var(::ReshapedDist, x) = x

@inline MeasureBase.vartransform_origin(ν::ReshapedDist) = unshaped(ν)
@inline MeasureBase.from_origin(ν::ReshapedDist, x) = varshape(ν)(x)
@inline MeasureBase.to_origin(ν::ReshapedDist, y) = unshaped(y, varshape(ν))
