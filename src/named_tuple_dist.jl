# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

_ntd_dist_and_shape(d::Distribution) = (d, varshape(d))

_ntd_dist_and_shape(s::ConstValueShape) = (ConstValueDist(s.value), s)

_ntd_dist_and_shape(s::IntervalSets.AbstractInterval) = _ntd_dist_and_shape(Uniform(minimum(s), maximum(s)))
_ntd_dist_and_shape(xs::AbstractVector{<:IntervalSets.AbstractInterval}) = _ntd_dist_and_shape(Product((s -> Uniform(minimum(s), maximum(s))).(xs)))
_ntd_dist_and_shape(xs::AbstractVector{<:Distribution}) = _ntd_dist_and_shape(product_distribution(xs))
_ntd_dist_and_shape(nt::NamedTuple) = _ntd_dist_and_shape(NamedTupleDist(nt))
_ntd_dist_and_shape(x::Number) = _ntd_dist_and_shape(ConstValueShape(x))
_ntd_dist_and_shape(x::AbstractArray{<:Number}) = _ntd_dist_and_shape(ConstValueShape(x))


"""
    NamedTupleDist <: Distribution

A distribution with `NamedTuple`-typed variates.

`NamedTupleDist` provides an effective mechanism to specify the distribution
of each variable/parameter in a set of named variables/parameters.

Calling `varshape` on a `NamedTupleDist` will yield a
[`NamedTupleShape`](@ref).

Constructors:

    NamedTupleDist(name1 = dist1, name2 = dist2, ...)
    NamedTupleDist(named_dists::NamedTuple)

Distributions may be given as `Distribution`s, intervals (uniform
distributions), constant values and nested `NamedTuple`s or
`NamedTupleDist`s:

```julia
d = NamedTupleDist(a = Normal(), b = (c = 0..1, d = MvNormal(I(2))), e = 42)
x = rand(d)
x.b.c isa Real
cov(unshaped(d))
```
"""
struct NamedTupleDist{
    names,
    DT <: (NTuple{N,Distribution} where N),
    AT <: (NTuple{N,ValueShapes.ValueAccessor} where N),
    VT
} <: Distribution{NamedTupleVariate{names},Continuous}
    _internal_distributions::NamedTuple{names,DT}
    _internal_shape::NamedTupleShape{names,AT,VT}
end 

export NamedTupleDist


function NamedTupleDist(::Type{VT}, dists::NamedTuple{names}) where {VT,names}
    dsb = map(_ntd_dist_and_shape, dists)
    NamedTupleDist(
        map(x -> x[1], dsb),
        NamedTupleShape(VT, map(x -> x[2], dsb))
    )
end

NamedTupleDist(dists::NamedTuple) = NamedTupleDist(NamedTuple, dists)

@inline NamedTupleDist(::Type{VT} ;named_dists...) where VT = NamedTupleDist(VT, values(named_dists))
@inline NamedTupleDist(;named_dists...) = NamedTupleDist(NamedTuple, values(named_dists))


@inline Base.convert(::Type{NamedTupleDist}, named_dists::NamedTuple) = NamedTupleDist(;named_dists...)


@inline _distributions(d::NamedTupleDist) = getfield(d, :_internal_distributions)
@inline _shape(d::NamedTupleDist) = getfield(d, :_internal_shape)

function Base.show(io::IO, d::NamedTupleDist)
    print(io, Base.typename(typeof(d)).name, "(")
    show(io, _distributions(d))
    print(io, ")")
end

@inline Base.keys(d::NamedTupleDist) = keys(_distributions(d))

@inline Base.values(d::NamedTupleDist) = values(_distributions(d))

@inline Base.getindex(d::NamedTupleDist, k::Symbol) = _distributions(d)[k]

@inline function Base.getproperty(d::NamedTupleDist, s::Symbol)
    # Need to include internal fields of NamedTupleShape to make Zygote happy (ToDo: still true?):
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
Base.merge(a::NamedTupleDist{names,DT,AT,VT}, b::NamedTupleDist, cs::NamedTupleDist...) where {names,DT,AT,VT} = 
    merge(NamedTupleDist(VT; a..., b...), cs...)

function Base.merge(a::NamedTupleDist{names,DT,AT,VT}, b::Union{NamedTupleDist,NamedTuple}, cs::Union{NamedTupleDist,NamedTuple}...) where {names,DT,AT,VT}
    merge(a, convert(NamedTupleDist, b), map(x -> convert(NamedTupleDist, x), cs)...)
end

varshape(d::NamedTupleDist) = _shape(d)


struct UnshapedNTD{NTD<:NamedTupleDist} <: Distribution{Multivariate,Continuous}
    shaped::NTD
end 


_ntd_length(d::Distribution) = length(d)
_ntd_length(d::ConstValueDist) = 0
_ntd_length(d::NamedTupleDist) = totalndof(varshape(d))

function Base.length(ud::UnshapedNTD)
    d = ud.shaped
    len = isempty(values(d)) ? 0 : sum(_ntd_length, values(d))
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

_ntd_logpdf(dist::NamedTupleDist, acc::ValueShapes.ValueAccessor, x::AbstractVector{<:Real}) =
    _ntd_logpdf(dist, _flat_view(x, acc))

# Sum of component contributions, zero for empty named tuples like for constants:
_ntd_sum(xs::Tuple) = sum(xs)
_ntd_sum(::Tuple{}) = zero(Float32)

function _ntd_logpdf(d::NamedTupleDist, x::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    _ntd_sum(map((dist, acc) -> _ntd_logpdf(dist, acc, x), distributions, accessors))
end


# ConstValueDist has no dof, so NamedTupleDist logpdf contribution must be zero:
_ntd_logpdf(dist::ConstValueDist, x::Any) = zero(Float32)

_ntd_logpdf(dist::Distribution, x::Any) = logpdf(dist, x)

function _ntd_logpdf(d::NamedTupleDist{names}, x::NamedTuple{names}) where names
    distributions = values(d)
    parvalues = values(x)
    _ntd_sum(map((dist, d) -> _ntd_logpdf(dist, d), distributions, parvalues))
end

Distributions.logpdf(d::NamedTupleDist{names}, x::NamedTuple{names}) where names = _ntd_logpdf(d, x)
Distributions.pdf(d::NamedTupleDist{names}, x::NamedTuple{names}) where names = exp(logpdf(d, x))

Distributions._logpdf(ud::UnshapedNTD, x::AbstractVector{<:Real}) = _ntd_logpdf(ud.shaped, x)

Distributions.logpdf(d::NamedTupleDist{names}, x::ShapedAsNT{names}) where names = _ntd_logpdf(d, convert(NamedTuple, x))
Distributions.pdf(d::NamedTupleDist{names}, x::ShapedAsNT{names}) where names = exp(logpdf(d, convert(NamedTuple, x)))


function _ntd_insupport(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    x::AbstractVector{<:Real}
)
    insupport(dist, float(x[acc]))
end

_ntd_insupport(dist::NamedTupleDist, acc::ValueShapes.ValueAccessor, x::AbstractVector{<:Real}) =
    _ntd_insupport(dist, _flat_view(x, acc))

function _ntd_insupport(d::NamedTupleDist, x::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    all(map((dist, acc) -> _ntd_insupport(dist, acc, x), distributions, accessors))
end


# ConstValueDist has no dof, set NamedTupleDist insupport contribution to true:
_ntd_insupport(dist::ConstValueDist, x::Any) = true

_ntd_insupport(dist::Distribution, x::Any) = insupport(dist, x)

function _ntd_insupport(d::NamedTupleDist{names}, x::NamedTuple{names}) where names
    distributions = values(d)
    parvalues = values(x)
    all(map((dist, d) -> _ntd_insupport(dist, d), distributions, parvalues))
end

Distributions.insupport(d::NamedTupleDist{names}, x::NamedTuple{names}) where names = _ntd_insupport(d, x)

Distributions.insupport(d::NamedTupleDist{names}, x::ShapedAsNT{names}) where names = _ntd_insupport(d, convert(NamedTuple, x))

Distributions.insupport(ud::UnshapedNTD, x::AbstractVector{<:Real}) = _ntd_insupport(ud.shaped, x)


function _ntd_rand!(
    rng::AbstractRNG, dist::ConstValueDist,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    x::AbstractVector{<:Real}
)
    nothing
end

function _ntd_rand!(
    rng::AbstractRNG, dist::Distribution{Univariate},
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
    rng::AbstractRNG, dist::Union{Distribution{<:PlainVariate}},
    acc::ValueShapes.ValueAccessor,
    x::AbstractVector{<:Real}
)
    rand!(rng, dist, view(x, acc))
    nothing
end

function _ntd_rand!(
    rng::AbstractRNG, dist::NamedTupleDist,
    acc::ValueShapes.ValueAccessor,
    x::AbstractVector{<:Real}
)
    _ntd_rand!(rng, dist, _flat_view(x, acc))
    nothing
end

function _ntd_rand!(rng::AbstractRNG, d::NamedTupleDist, x::AbstractVector{<:Real})
    distributions = values(d)
    accessors = values(varshape(d))
    map((dist, acc) -> _ntd_rand!(rng, dist, acc, x), distributions, accessors)
    x
end

@inline Distributions._rand!(rng::AbstractRNG, ud::UnshapedNTD, x::AbstractVector{<:Real}) = _ntd_rand!(rng, ud.shaped, x)


# Statistics f (mode, mean, var) of the components, constants yield nothing:
_ntd_stat(f, dist::ConstValueDist) = nothing
_ntd_stat(f, dist::Distribution) = f(dist)
_ntd_stat(f, dist::NamedTupleDist) = NamedTuple{keys(_distributions(dist))}(_ntd_stats(f, values(_distributions(dist))))

# Workaround, Distributions.jl doesn't define mode for Product:
_ntd_stat(::typeof(mode), dist::Distributions.Product) = map(mode, dist.v)

_ntd_stats(f, ::Tuple{}) = ()
_ntd_stats(f, dists::Tuple) = (_ntd_stat(f, first(dists)), _ntd_stats(f, Base.tail(dists))...)


_stat_realtype(::Nothing) = Bool
_stat_realtype(x::Real) = typeof(x)
_stat_realtype(x::AbstractArray{<:Real}) = eltype(x)
_stat_realtype(x::NamedTuple) = _stats_realtype(values(x))

_stats_realtype(::Tuple{}) = Bool
_stats_realtype(stats::Tuple) = promote_type(_stat_realtype(first(stats)), _stats_realtype(Base.tail(stats)))


# Writes component statistics into the flat parameter vector:
_ntd_write!(x::AbstractVector{<:Real}, ::Nothing, acc::ValueShapes.ValueAccessor) = nothing

function _ntd_write!(x::AbstractVector{<:Real}, stat, acc::ValueShapes.ValueAccessor)
    view(x, acc) .= stat
    nothing
end

_ntd_write!(x::AbstractVector{<:Real}, stats::NamedTuple, acc::NamedTupleAccessor) =
    _ntd_write!(_flat_view(x, acc), stats, acc.shape)

function _ntd_write!(x::AbstractVector{<:Real}, stats::NamedTuple, vs::NamedTupleShape)
    map((stat, acc) -> _ntd_write!(x, stat, acc), values(stats), values(vs))
    nothing
end


function _ntd_fill(f, d::NamedTupleDist)
    stats = _ntd_stat(f, d)
    x = Vector{_stat_realtype(stats)}(undef, varshape(d))
    _ntd_write!(x, stats, varshape(d))
    x
end


StatsBase.mode(d::NamedTupleDist) = varshape(d)(mode(unshaped(d)))

StatsBase.mode(ud::UnshapedNTD) = _ntd_fill(mode, ud.shaped)


Statistics.mean(d::NamedTupleDist) = varshape(d)(mean(unshaped(d)))

Statistics.mean(ud::UnshapedNTD) = _ntd_fill(mean, ud.shaped)


Statistics.var(d::NamedTupleDist) = variance_shape(varshape(d))(var(unshaped(d)))

Statistics.var(ud::UnshapedNTD) = _ntd_fill(var, ud.shaped)


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

function _ntd_cov!(
    dist::NamedTupleDist,
    acc::ValueShapes.ValueAccessor,
    A_cov::AbstractMatrix{<:Real}
)
    idxs1, idxs2 = map(ax -> view_range(ax, acc), axes(A_cov))
    _ntd_cov!(view(A_cov, idxs1, idxs2), dist)
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
