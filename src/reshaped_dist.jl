# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


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
_variate_form(shape::ArrayShape{T,1}) where T = Multivariate
_variate_form(shape::ArrayShape{T,2}) where T = Matrixvariate
_variate_form(shape::NamedTupleShape{names}) where names = NamedTupleVariate{names}

_with_zeroconst(shape::AbstractValueShape) = replace_const_shapes(const_zero_shape, shape)


function ReshapedDist(dist::MultivariateDistribution{VS}, shape::AbstractValueShape) where {VS}
    @argcheck totalndof(varshape(dist)) == totalndof(shape)
    VF = _variate_form(shape)
    D = typeof(dist)
    S = typeof(shape)
    ReshapedDist{VF,VS,D,S}(dist, shape)
end


@static if VERSION >= v"1.3"
    (shape::AbstractValueShape)(dist::MultivariateDistribution) = ReshapedDist(dist, shape)
else
    (shape::ScalarShape)(dist::MultivariateDistribution) = ReshapedDist(dist, shape)
    (shape::ArrayShape)(dist::MultivariateDistribution) = ReshapedDist(dist, shape)
    (shape::ConstValueShape)(dist::MultivariateDistribution) = ReshapedDist(dist, shape)
    (shape::NamedTupleShape)(dist::MultivariateDistribution) = ReshapedDist(dist, shape)
end

function (shape::ArrayShape{T,1})(dist::MultivariateDistribution) where T
    @argcheck totalndof(varshape(dist)) == totalndof(shape)
    dist
end


@inline varshape(rd::ReshapedDist) = rd.shape

@inline unshaped(rd::ReshapedDist) = rd.dist


Random.rand(rng::AbstractRNG, rd::ReshapedDist) = stripscalar(rand(rng, rd, ()))
Random.rand(rng::AbstractRNG, rd::ReshapedDist, ::Tuple{}) = varshape(rd)(rand(rng, unshaped(rd)))
Random.rand(rng::AbstractRNG, rd::ReshapedDist, n::Int) = varshape(rd).(nestedview(rand(rng, unshaped(rd), n)))
Random.rand(rng::AbstractRNG, rd::ReshapedDist, dims::Dims{1}) = rand(rng, rd, dims[1])


Statistics.mean(rd::ReshapedDist) = stripscalar(varshape(rd)(mean(unshaped(rd))))

StatsBase.mode(rd::ReshapedDist) = stripscalar(varshape(rd)(mode(unshaped(rd))))

Statistics.var(rd::ReshapedDist) = stripscalar(_with_zeroconst(varshape(rd))(var(unshaped(rd))))

Statistics.cov(rd::ReshapedDist{Multivariate}) = cov(unshaped(rd))
