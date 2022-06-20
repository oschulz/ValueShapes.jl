# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    const NamedTupleVariate{names} = StructVariate{NamedTuple{names}}

Variate kind for `NamedTuple`s.
"""
const NamedTupleVariate{names} = StructVariate{NamedTuple{names}}  # ToDo: Use StructVariate{<:NamedTuple{names}} instead?

Distributions.insupport(d::ConstValueDist{<:NamedTupleVariate{names}}, x::NamedTuple{names}) where names = x == d.value
Distributions.pdf(d::ConstValueDist{<:NamedTupleVariate{names}}, x::NamedTuple{names}) where names = _pdf_impl(d, x)
Distributions.logpdf(d::ConstValueDist{<:NamedTupleVariate{names}}, x::NamedTuple{names}) where names = log(pdf(d, x))


function _rand_flat_impl(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}) where names
    shape = varshape(d)
    X = Vector{default_unshaped_eltype(shape)}(undef, totalndof(varshape(d)))
    (shape, rand!(rng, unshaped(d), X))
end

function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}) where names
    shape, X = _rand_flat_impl(rng, d)
    shape(X)
end

function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, dims::Tuple{}) where names
    shape, X = _rand_flat_impl(rng, d)
    shape.(Fill(X))
end


function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, dims::Dims) where names
    shape = varshape(d)
    X_flat = Array{default_unshaped_eltype(shape)}(undef, totalndof(varshape(d)), dims...)
    X = ArrayOfSimilarVectors(X_flat)
    rand!(rng, unshaped(d), X)
    shape.(X)
end


function Random.rand!(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    rand!(Random.default_rng(), d, x)
end

function Random.rand!(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    valshape(x) >= varshape(d) || throw(ArgumentError("Shapes of variate and value are not compatible"))
    rand!(rng, unshaped(d), unshaped(x))
    x
end


function _aov_rand_impl!(rng::AbstractRNG, d::Distribution{Multivariate}, X::ArrayOfSimilarVectors{<:Real})
    rand!(rng, unshaped(d), flatview(X))
end

# Workaround for current limitations of ArraysOfArrays.unshaped for standard arrays of vectors
function _aov_rand_impl!(rng::AbstractRNG, d::Distribution{Multivariate}, X::AbstractArray{<:AbstractVector{<:Real}})
    rand!.(Ref(rng), Ref(unshaped(d)), X)
end

function Random.rand!(rng::AbstractRNG, d::Distribution{<:NamedTupleVariate}, X::ShapedAsNTArray)
    elshape(X) >= varshape(d) || throw(ArgumentError("Shapes of variate and value are not compatible"))
    _aov_rand_impl!(rng, unshaped(d), unshaped.(X))
    X
end


function Distributions.logpdf(d::Distribution{NamedTupleVariate{names}}, x::NamedTuple{names}) where names
    logpdf(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.logpdf(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    @argcheck valshape(x) <= varshape(d)
    logpdf(unshaped(d), unshaped(x))
end

function Distributions.logpdf(d::Distribution{NamedTupleVariate{names}}, x::AbstractArray{<:NamedTuple{names},0}) where names
    logpdf(unshaped(d), unshaped(x, varshape(d)))
end


function Distributions.pdf(d::Distribution{NamedTupleVariate{names}}, x::NamedTuple{names}) where names
    pdf(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.pdf(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    @argcheck valshape(x) <= varshape(d)
    pdf(unshaped(d), unshaped(x))
end

function Distributions.pdf(d::Distribution{NamedTupleVariate{names}}, x::AbstractArray{<:NamedTuple{names},0}) where names
    pdf(unshaped(d), unshaped(x, varshape(d)))
end


function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, x::NamedTuple{names}) where names
    insupport(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    @argcheck valshape(x) <= varshape(d)
    insupport(unshaped(d), unshaped(x))
end

function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, x::AbstractArray{<:NamedTuple{names},0}) where names
    insupport(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, X::AbstractArray{<:NamedTuple{names},N}) where {N,names}
    Distributions.insupport!(BitArray(undef, size(X)), d, X)
end

function Distributions.insupport!(r::AbstractArray{Bool,N}, d::Distribution{NamedTupleVariate{names}}, X::AbstractArray{<:NamedTuple{names},N}) where {N,names}
    r .= insupport.(Ref(d), X)
end



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


#function Base.show(io::IO, d::NamedTupleDist)
#    print(io, Base.typename(typeof(d)).name, "(")
#    show(io, _distributions(d))
#    print(io, ")")
#end

function Base.show(io::IO, d::NamedTupleDist)
    print(io, Base.typename(typeof(d)).name, "{")
    show(io, propertynames(d))
    print(io, "}(…)")
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


MeasureBase.getdof(d::NamedTupleDist) = sum(map(MeasureBase.getdof, values(d)))

# Bypass `checked_arg`, would require potentially costly transformation:
@inline MeasureBase.checked_arg(::NamedTupleDist, x) = x

@inline MeasureBase.transport_origin(ν::NamedTupleDist) = unshaped(ν)
@inline MeasureBase.from_origin(ν::NamedTupleDist, x) = varshape(ν)(x)
@inline MeasureBase.to_origin(ν::NamedTupleDist, y) = unshaped(y, varshape(ν))



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

Distributions.logpdf(d::NamedTupleDist{names}, x::ShapedAsNT{names}) where names = _ntd_logpdf(d, convert(NamedTuple, x))
Distributions.pdf(d::NamedTupleDist{names}, x::ShapedAsNT{names}) where names = exp(logpdf(d, convert(NamedTuple, x)))


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
    rng::AbstractRNG, dist::Union{Distribution{<:ArrayLikeVariate}},
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
StatsBase.mode(d::NamedTupleDist) = varshape(d)(mode(unshaped(d)))

StatsBase.mode(ud::UnshapedNTD) = _ntd_mode(ud.shaped)


_ntd_mean(dist::ConstValueDist) = Float32[]
_ntd_mean(dist::Distribution) = mean(unshaped(dist))

# ToDo/Decision: Return NamedTuple or ShapedAsNT?
Statistics.mean(d::NamedTupleDist) = varshape(d)(mean(unshaped(d)))

function Statistics.mean(ud::UnshapedNTD)
    d = ud.shaped
    vcat(map(d -> _ntd_mean(d), values(ValueShapes._distributions(d)))...)
end


_ntd_var(dist::ConstValueDist) = Float32[]
_ntd_var(dist::Distribution) = var(unshaped(dist))

# ToDo/Decision: Return NamedTuple or ShapedAsNT?
Statistics.var(d::NamedTupleDist) = variance_shape(varshape(d))(var(unshaped(d)))

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


MeasureBase.getdof(d::ValueShapes.UnshapedNTD) = getdof(d.shaped)



# ToDo add custom rrules for parts of the NamedTupleDist transform process.

_flat_ntd_elshape(d::Distribution) = ArrayShape{Real}(getdof(d))

function _flat_ntd_accessors(d::NamedTupleDist{names,DT,AT,VT}) where {names,DT,AT,VT}
    shapes = map(_flat_ntd_elshape, values(d))
    vs = NamedTupleShape(VT, NamedTuple{names}(shapes))
    values(vs)
end


function _flat_ntdistelem_to_stdmv(trg::UvStdMeasure, sd::Distribution, x_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor)
    # ToDo: This may fail for Dirichlet and the like, trg_acc may be wrong:
    transport_def(mv_stdmeasure(trg, trg_acc.len), unshaped(sd), trg_acc(x_unshaped))
end

function _flat_ntdistelem_to_stdmv(trg::MvStdMeasure, sd::ConstValueDist, x_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor)
    Zeros{Bool}(0)
end

function MeasureBase.transport_def(ν::UvStdMeasure, μ::ValueShapes.UnshapedNTD, x::AbstractVector{<:Real})
    # @argcheck length(src) == length(eachindex(x))
    trg_accessors = _flat_ntd_accessors(μ.shaped)
    rs = map((acc, sd) -> _flat_ntdistelem_to_stdmv(uv_stdmeasure(ν), sd, x, acc), trg_accessors, values(μ.shaped))
    vcat(rs...)
end


function _stdmv_to_flat_ntdistelem(td::Distribution, src::UvStdMeasure, x::AbstractVector{<:Real}, src_acc::ValueAccessor)
    sd = resize_mvstdmeasure(src, src_acc.len)
    transport_def(unshaped(td), sd, src_acc(x))
end

function _stdmv_to_flat_ntdistelem(td::ConstValueDist, src::MvStdMeasure, x::AbstractVector{<:Real}, src_acc::ValueAccessor)
    Zeros{Bool}(0)
end

function MeasureBase.transport_def(ν::ValueShapes.UnshapedNTD, μ::MvStdMeasure, x::AbstractVector{<:Real})
    # @argcheck length(μ) == length(eachindex(x))
    src_accessors = _flat_ntd_accessors(ν.shaped)
    rs = map((acc, td) -> _stdmv_to_flat_ntdistelem(td, uv_stdmeasure(μ), x, acc), src_accessors, values(ν.shaped))
    vcat(rs...)
end


