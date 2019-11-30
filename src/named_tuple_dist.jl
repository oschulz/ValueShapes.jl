# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

_ntd_dist_and_shape(d::Distribution) = (d, varshape(d))

_ntd_dist_and_shape(s::ConstValueShape) = (ConstValueDist(s.value), s)

_ntd_dist_and_shape(s::IntervalSets.AbstractInterval) = _ntd_dist_and_shape(Uniform(minimum(s), maximum(s)))
_ntd_dist_and_shape(xs::AbstractVector{<:IntervalSets.AbstractInterval}) = _ntd_dist_and_shape(Product((s -> Uniform(minimum(s), maximum(s))).(xs)))
_ntd_dist_and_shape(xs::AbstractVector{<:Distribution}) = _ntd_dist_and_shape(Product(xs))
_ntd_dist_and_shape(x::Number) = _ntd_dist_and_shape(ConstValueShape(x))
_ntd_dist_and_shape(x::AbstractArray{<:Number}) = _ntd_dist_and_shape(ConstValueShape(x))


"""
    NamedTupleDist <: MultivariateDistribution
    NamedTupleDist <: MultivariateDistribution

A distribution with `NamedTuple`-typed variates.

Can be used to describe the distribution of each parameter in a set of
named parameters. If the distribution is used as a Bayesian prior,
the `NamedTupleDist` then specifies the prior on each named parameter.

Calling `varshape` on a `NamedTupleDist` will yield a
[`NamedTupleShape`](@ref).
"""
struct NamedTupleDist{
    names,
    DT <: (NTuple{N,Distribution} where N),
    AT <: (NTuple{N,ValueShapes.ValueAccessor} where N),
} <: Distribution{Multivariate,Continuous}
    _internal_distributions::NamedTuple{names,DT}
    _internal_shapes::NamedTupleShape{names,AT}
end 

export NamedTupleDist


function NamedTupleDist(param_priors::NamedTuple{names}) where {names}
    dsb = map(_ntd_dist_and_shape, param_priors)
    NamedTupleDist(
        map(x -> x[1], dsb),
        NamedTupleShape(map(x -> x[2], dsb))
    )
end

@inline NamedTupleDist(;named_priors...) = NamedTupleDist(values(named_priors))



@inline _distributions(d::NamedTupleDist) = getfield(d, :_internal_distributions)
@inline _shapes(d::NamedTupleDist) = getfield(d, :_internal_shapes)


@inline Base.keys(d::NamedTupleDist) = keys(_distributions(d))

@inline Base.values(d::NamedTupleDist) = values(_distributions(d))

@inline function Base.getproperty(d::NamedTupleDist, s::Symbol)
    # Need to include internal fields of NamedTupleShape to make Zygote happy:
    if s == :_internal_distributions
        getfield(d, :_internal_distributions)
    elseif s == :_shapes
        getfield(d, :_internal_shapes)
    else
        getproperty(_distributions(d), s)
    end
end

@inline function Base.propertynames(d::NamedTupleDist, private::Bool = false)
    names = propertynames(_distributions(d))
    if private
        (names..., :_internal_distributions, :_internal_shapes)
    else
        names
    end
end


ValueShapes.varshape(d::NamedTupleDist) = _shapes(d)



_ntd_length(d::Distribution) = length(d)
_ntd_length(d::ConstValueDist) = 0

Base.length(d::NamedTupleDist) = sum(_ntd_length, values(d))


function _ntd_logpdf(
    dist::ConstValueDist,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    params::AbstractVector{<:Real}
)
    float(zero(eltype(params)))
end

function _ntd_logpdf(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    params::AbstractVector{<:Real}
)
    logpdf(dist, float(params[acc]))
end

function Distributions.logpdf(d::NamedTupleDist, params::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    sum(map((dist, acc) -> _ntd_logpdf(dist, acc, params), distributions, accessors))
end


# ConstValueDist has no dof, so NamedTupleDist logpdf contribution must be zero:
_ntd_logpdf(dist::ConstValueDist, params::Any) = zero(Float32)

_ntd_logpdf(dist::Distribution, params::Any) = logpdf(dist, params)

function Distributions.logpdf(d::NamedTupleDist{names}, params::NamedTuple{names}) where names
    distributions = values(d)
    parvalues = values(params)
    sum(map((dist, d) -> _ntd_logpdf(dist, d), distributions, parvalues))
end


function _ntd_rand!(
    rng::AbstractRNG, dist::ConstValueDist,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    params::AbstractVector{<:Real}
)
    nothing
end

function _ntd_rand!(
    rng::AbstractRNG, dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    params::AbstractVector{<:Real}
)
    rand!(rng, dist, view(params, acc))
    nothing
end

function Distributions._rand!(rng::AbstractRNG, d::NamedTupleDist, params::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    map((dist, acc) -> _ntd_rand!(rng, dist, acc, params), distributions, accessors)
    params
end

#Random.rand(rng::AbstractRNG, d::NamedTupleDist) = rand!(rng, d, Vector{Float64}(undef, length(d)))


function _ntd_var_or_cov!(A_cov::AbstractArray{<:Real,0}, dist::Distribution{Univariate})
    A_cov[] = var(dist)
    nothing
end

function _ntd_var_or_cov!(A_cov::AbstractArray{<:Real,2}, dist::Distribution{Multivariate})
    A_cov[:, :] = cov(dist)
    nothing
end

function _ntd_cov!(
    dist::ConstValueDist,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    A_cov::AbstractMatrix{<:Real}
)
    nothing
end

function _ntd_cov!(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    A_cov::AbstractMatrix{<:Real}
)
    _ntd_var_or_cov!(view(A_cov, acc, acc), dist)
    nothing
end

function Statistics.cov(d::NamedTupleDist)
    let n = length(d), A_cov = zeros(n, n)
        distributions = values(d)
        accessors = values(varshape(d))
        map((dist, acc) -> _ntd_cov!(dist, acc, A_cov), distributions, accessors)
        A_cov
    end
end
