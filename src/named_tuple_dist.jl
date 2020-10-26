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

`NamedTupleDist` provides an effective mechanism to specify the distribution
of each variable/parameter in a set of named variables/parameters.

Calling `varshape` on a `NamedTupleDist` will yield a
[`NamedTupleShape`](@ref).
"""
struct NamedTupleDist{
    names,
    DT <: (NTuple{N,Distribution} where N),
    AT <: (NTuple{N,ValueShapes.ValueAccessor} where N),
} <: Distribution{NamedTupleVariate{names},Continuous}
    _internal_distributions::NamedTuple{names,DT}
    _internal_shape::NamedTupleShape{names,AT}
end 

export NamedTupleDist


function NamedTupleDist(dists::NamedTuple{names}) where {names}
    dsb = map(_ntd_dist_and_shape, dists)
    NamedTupleDist(
        map(x -> x[1], dsb),
        NamedTupleShape(map(x -> x[2], dsb))
    )
end

@inline NamedTupleDist(;named_dists...) = NamedTupleDist(values(named_dists))



@inline _distributions(d::NamedTupleDist) = getfield(d, :_internal_distributions)
@inline _shape(d::NamedTupleDist) = getfield(d, :_internal_shape)


@inline Base.keys(d::NamedTupleDist) = keys(_distributions(d))

@inline Base.values(d::NamedTupleDist) = values(_distributions(d))

@inline function Base.getproperty(d::NamedTupleDist, s::Symbol)
    # Need to include internal fields of NamedTupleShape to make Zygote happy:
    if s == :_internal_distributions
        getfield(d, :_internal_distributions)
    elseif s == :_internal_shape
        getfield(d, :_internal_shape)
    else
        getproperty(_distributions(d), s)
    end
end

@inline function Base.propertynames(d::NamedTupleDist, private::Bool = false)
    names = propertynames(_distributions(d))
    if private
        (names..., :_internal_distributions, :_internal_shape)
    else
        names
    end
end


@inline Base.map(f, dist::NamedTupleDist) = map(f, _distributions(dist))


Base.merge(a::NamedTuple, dist::NamedTupleDist{names}) where {names} = merge(a, _distributions(dist))
Base.merge(a::NamedTupleDist) = a
Base.merge(a::NamedTupleDist, b::NamedTupleDist, cs::NamedTupleDist...) = merge(NamedTupleDist(;a..., b...), cs...)

varshape(d::NamedTupleDist) = _shape(d)


struct UnshapedNTD{NTD<:NamedTupleDist} <: Distribution{Multivariate,Continuous}
    shaped::NTD
end 


_ntd_length(d::Distribution) = length(d)
_ntd_length(d::ConstValueDist) = 0

function Base.length(ud::UnshapedNTD)
    d = ud.shaped
    len = sum(_ntd_length, values(d))
    @assert len == totalndof(varshape(d))
    len
end

Base.eltype(ud::UnshapedNTD) = default_unshaped_eltype(varshape(ud.shaped))


unshaped(d::NamedTupleDist) = UnshapedNTD(d)


function _ntd_logpdf(
    dist::ConstValueDist,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    x::AbstractVector{<:Real}
)
    float(zero(eltype(x)))
end

function _ntd_logpdf(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    x::AbstractVector{<:Real}
)
    logpdf(dist, float(x[acc]))
end

function _ntd_logpdf(d::NamedTupleDist, x::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    sum(map((dist, acc) -> _ntd_logpdf(dist, acc, x), distributions, accessors))
end


# ConstValueDist has no dof, so NamedTupleDist logpdf contribution must be zero:
_ntd_logpdf(dist::ConstValueDist, x::Any) = zero(Float32)

_ntd_logpdf(dist::Distribution, x::Any) = logpdf(dist, x)

function _ntd_logpdf(d::NamedTupleDist{names}, x::NamedTuple{names}) where names
    distributions = values(d)
    parvalues = values(x)
    sum(map((dist, d) -> _ntd_logpdf(dist, d), distributions, parvalues))
end

Distributions.logpdf(d::NamedTupleDist{names}, x::NamedTuple{names}) where names = _ntd_logpdf(d, x)
Distributions.pdf(d::NamedTupleDist{names}, x::NamedTuple{names}) where names = exp(logpdf(d, x))

Distributions._logpdf(ud::UnshapedNTD, x::AbstractVector{<:Real}) = _ntd_logpdf(ud.shaped, x)


function _ntd_insupport(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    x::AbstractVector{<:Real}
)
    insupport(dist, float(x[acc]))
end

function _ntd_insupport(d::NamedTupleDist, x::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    prod(map((dist, acc) -> _ntd_insupport(dist, acc, x), distributions, accessors))
end


# ConstValueDist has no dof, set NamedTupleDist insupport contribution to true:
_ntd_insupport(dist::ConstValueDist, x::Any) = true

_ntd_insupport(dist::Distribution, x::Any) = insupport(dist, x)

function _ntd_insupport(d::NamedTupleDist{names}, x::NamedTuple{names}) where names
    distributions = values(d)
    parvalues = values(x)
    prod(map((dist, d) -> _ntd_insupport(dist, d), distributions, parvalues))
end

Distributions.insupport(d::NamedTupleDist{names}, x::NamedTuple{names}) where names = _ntd_insupport(d, x)

Distributions.insupport(ud::UnshapedNTD, x::AbstractVector{<:Real}) = _ntd_insupport(ud.shaped, x)


function _ntd_rand!(
    rng::AbstractRNG, dist::ConstValueDist,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    x::AbstractVector{<:Real}
)
    nothing
end

function _ntd_rand!(
    rng::AbstractRNG, dist::UnivariateDistribution,
    acc::ValueShapes.ValueAccessor,
    x::AbstractVector{<:Real}
)
    x_view = view(x, acc)
    idxs = eachindex(x_view)
    @assert length(idxs) == 1
    x_view[first(idxs)] = rand(rng, dist)
    nothing
end

function _ntd_rand!(
    rng::AbstractRNG, dist::MultivariateDistribution,
    acc::ValueShapes.ValueAccessor,
    x::AbstractVector{<:Real}
)
    rand!(rng, dist, view(x, acc))
    nothing
end

function _ntd_rand!(rng::AbstractRNG, d::NamedTupleDist, x::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    map((dist, acc) -> _ntd_rand!(rng, dist, acc, x), distributions, accessors)
    x
end

@inline Distributions._rand!(rng::AbstractRNG, ud::UnshapedNTD, x::AbstractVector{<:Real}) = _ntd_rand!(rng, ud.shaped, x)


function _ntd_mode!(
    dist::ConstValueDist,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    params::AbstractVector{<:Real}
)
    nothing
end

function _ntd_mode!(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    params::AbstractVector{<:Real}
)
    view(params, acc) .= mode(dist)
    nothing
end

# Workaround, Distributions.jl doesn't define mode for Product:
function _ntd_mode!(
    dist::Distributions.Product,
    acc::ValueShapes.ValueAccessor,
    params::AbstractVector{<:Real}
)
    view(params, acc) .= map(mode, dist.v)
    nothing
end

function _ntd_mode!(x::AbstractVector{<:Real}, d::NamedTupleDist)
    distributions = values(d)
    shape = varshape(d)
    accessors = values(shape)
    map((dist, acc) -> _ntd_mode!(dist, acc, x), distributions, accessors)
    nothing
end

function _ntd_mode(d::NamedTupleDist)
    x = Vector{default_unshaped_eltype(varshape(d))}(undef,varshape(d))
    _ntd_mode!(x, d)
    x
end


# ToDo/Decision: Return NamedTuple or ShapedAsNT?
StatsBase.mode(d::NamedTupleDist) = varshape(d)(_ntd_mode(d))[]

StatsBase.mode(ud::UnshapedNTD) = _ntd_mode(ud.shaped)


_ntd_var(dist::ConstValueDist) = Float32[]

_ntd_var(dist::Distribution) = var(unshaped(dist))


# ToDo/Decision: Return NamedTuple or ShapedAsNT?
Statistics.var(d::NamedTupleDist) = map(var, _distributions(d))

function Statistics.var(ud::UnshapedNTD)
    d = ud.shaped
    vcat(map(d -> _ntd_var(d), values(ValueShapes._distributions(d)))...)
end


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

function _ntd_cov!(A_cov::AbstractMatrix{<:Real}, d::NamedTupleDist)
    distributions = values(d)
    accessors = values(varshape(d))
    map((dist, acc) -> _ntd_cov!(dist, acc, A_cov), distributions, accessors)
    A_cov
end

function _ntd_cov(d::NamedTupleDist)
    n = totalndof(varshape(d))
    A_cov = zeros(n, n)
    _ntd_cov!(A_cov, d)
end


Statistics.cov(ud::UnshapedNTD) = _ntd_cov(ud.shaped)
