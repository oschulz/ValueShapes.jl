# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


@inline _varoffset_cumsum_impl(s, x, y, rest...) = (s, _varoffset_cumsum_impl(s+x, y, rest...)...)
@inline _varoffset_cumsum_impl(s,x) = (s,)
@inline _varoffset_cumsum_impl(s) = ()
@inline _varoffset_cumsum(x::Tuple) = _varoffset_cumsum_impl(0, x...)


"""
    NamedTupleShape{names,...} <: AbstractValueShape

Defines the shape of a `NamedTuple` (resp.  set of variables, parameters,
etc.).

Constructors:

    NamedTupleShape(name1 = shape1::AbstractValueShape, ...)
    NamedTupleShape(named_shapes::NamedTuple)

e.g.

    shape = NamedTupleShape(
        a = ArrayShape{Real}(2, 3),
        b = ScalarShape{Real}(),
        c = ArrayShape{Real}(4)
    )

Example:

```julia
shape = NamedTupleShape(
    a = ScalarShape{Real}(),
    b = ArrayShape{Real}(2, 3),
    c = ConstValueShape(42)
)
data = VectorOfSimilarVectors{Float64}(shape)
resize!(data, 10)
rand!(flatview(data))
table = shape.(data)
fill!(table.a, 4.2)
all(x -> x == 4.2, view(flatview(data), 1, :))
```

See also the documentation of [`AbstractValueShape`](@ref).
"""
struct NamedTupleShape{names,AT<:(NTuple{N,ValueAccessor} where N)} <: AbstractValueShape
    _accessors::NamedTuple{names,AT}
    _flatdof::Int

    @inline function NamedTupleShape(shape::NamedTuple{names,<:NTuple{N,AbstractValueShape}}) where {names,N}
        labels = keys(shape)
        shapes = values(shape)
        shapelengths = map(totalndof, shapes)
        offsets = _varoffset_cumsum(shapelengths)
        accessors = map(ValueAccessor, shapes, offsets)
        # acclengths = map(x -> x.len, accessors)
        # @assert shapelengths == acclengths
        n_flattened = sum(shapelengths)
        named_accessors = NamedTuple{labels}(accessors)
        new{names,typeof(accessors)}(named_accessors, n_flattened)
    end
end

export NamedTupleShape

@inline NamedTupleShape(;named_shapes...) = NamedTupleShape(values(named_shapes))


@inline _accessors(x::NamedTupleShape) = getfield(x, :_accessors)
@inline _flatdof(x::NamedTupleShape) = getfield(x, :_flatdof)

@inline totalndof(shape::NamedTupleShape) = _flatdof(shape)

@inline Base.keys(shape::NamedTupleShape) = keys(_accessors(shape))

@inline Base.values(shape::NamedTupleShape) = values(_accessors(shape))

@inline function Base.getproperty(shape::NamedTupleShape, p::Symbol)
    # Need to include internal fields of NamedTupleShape to make Zygote happy:
    if p == :_accessors
        getfield(shape, :_accessors)
    elseif p == :_flatdof
        getfield(shape, :_flatdof)
    else
        getproperty(_accessors(shape), p)
    end
end

@inline Base.propertynames(shape::NamedTupleShape) = propertynames(_accessors(shape))

@inline Base.length(shape::NamedTupleShape) = length(_accessors(shape))

@inline Base.getindex(shape::NamedTupleShape, i::Integer) = getindex(_accessors(shape), i)

@inline Base.map(f, shape::NamedTupleShape) = map(f, _accessors(shape))

function Base.merge(a::NamedTuple, shape::NamedTupleShape{names}) where {names}
    merge(a, NamedTuple{names}(map(x -> valshape(x), values(shape))))
end


valshape(x::NamedTuple) = NamedTupleShape(map(valshape, x))


(shape::NamedTupleShape)(::UndefInitializer) = map(x -> valshape(x)(undef), shape)


Base.@propagate_inbounds function (shape::NamedTupleShape)(data::AbstractVector{<:Real})
    @boundscheck _checkcompat(shape, data)
    accessors = _accessors(shape)
    map(va -> data[va], accessors)
end


@inline _multi_promote_type() = Nothing
@inline _multi_promote_type(T::Type) = T
@inline _multi_promote_type(T::Type, U::Type, rest::Type...) = promote_type(T, _multi_promote_type(U, rest...))


@inline nonabstract_eltype(shape::NamedTupleShape) =
    _multi_promote_type(map(nonabstract_eltype, values(shape))...)


Base.@propagate_inbounds function _bcasted_apply(shape::NamedTupleShape, data::AbstractVector{<:AbstractVector{<:Real}})
    accessors = _accessors(shape)
    cols = map(va -> getindex.(data, va), accessors)
    TypedTables.Table(cols)
end

# Specialize (::NamedTupleShape).(::AbstractVector{<:AbstractVector}):
Base.copy(instance::VSBroadcasted1{<:NamedTupleShape,AbstractVector{<:AbstractVector}}) =
    _bcasted_apply(instance.f, instance.args[1])    



# ToDo: Implement support for ValueAccessor{NamedTupleShape}
