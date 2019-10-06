# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


struct VariableDataAccessor{S<:AbstractValueShape}
    shape::S
    offset::Int
    len::Int

    VariableDataAccessor{S}(shape::S, offset::Int) where {S<:AbstractValueShape} =
        new{S}(shape, offset, totalndof(shape))
end

VariableDataAccessor(shape::S, offset::Int) where {S<:AbstractValueShape} = VariableDataAccessor{S}(shape, offset)


const ScalarAccessor{T} = VariableDataAccessor{ScalarShape{T}} where {T}
const ArrayAccessor{T,N} = VariableDataAccessor{ArrayShape{T,N}} where {T,N}
const ConstAccessor = VariableDataAccessor{ConstValueShape{T}} where {T}


nonabstract_eltype(va::VariableDataAccessor) = nonabstract_eltype(va.shape)

AbstractValueShape(va::VariableDataAccessor) = va.shape
Base.convert(::Type{AbstractValueShape}, va::VariableDataAccessor) = ValueShape(va)

Base.size(va::VariableDataAccessor) = size(va.shape)
Base.length(va::VariableDataAccessor) = length(va.shape)


# ToDo: implement Base.getindex for VariableDataAccessor? Allow only contiguous?





@inline function _view_range(idxs::AbstractUnitRange{<:Integer}, va::VariableDataAccessor)
    from = first(idxs) + va.offset
    to = from + va.len - 1
    from:to
end

@inline _view_idxs(idxs::AbstractUnitRange{<:Integer}, va::VariableDataAccessor) = _view_range(idxs, va)
@inline _view_idxs(idxs::AbstractUnitRange{<:Integer}, va::ScalarAccessor) = first(idxs) + va.offset


Base.@propagate_inbounds (va::ScalarAccessor)(data::AbstractVector) =
    data[_view_idxs(axes(data, 1), va)]

Base.@propagate_inbounds (va::ArrayAccessor{T,1})(data::AbstractVector) where {T} =
    view(data, _view_idxs(axes(data, 1), va))

Base.@propagate_inbounds (va::ArrayAccessor{T,N})(data::AbstractVector) where {T,N} =
    reshape(view(data, _view_idxs(axes(data, 1), va)), size(va.shape)...)

@inline (va::ConstAccessor)(::AbstractVector) = va.shape.value


Base.@propagate_inbounds function (va::ScalarAccessor)(data::AbstractVectorOfSimilarVectors)
    flat_data = flatview(data)
    idxs = _view_idxs(axes(flat_data, 1), va)
    view(flat_data, idxs, :)
end

Base.@propagate_inbounds function (va::ArrayAccessor{T,1})(data::AbstractVectorOfSimilarVectors) where {T,N}
    flat_data = flatview(data)
    idxs = _view_idxs(axes(flat_data, 1), va)
    fpview = view(flat_data, idxs, :)
    VectorOfSimilarVectors(fpview)
end

Base.@propagate_inbounds function (va::ArrayAccessor{T,N})(data::AbstractVectorOfSimilarVectors) where {T,N}
    flat_data = flatview(data)
    idxs = _view_idxs(axes(flat_data, 1), va)
    fpview = view(flat_data, idxs, :)
    VectorOfSimilarArrays(reshape(fpview, size(va.shape)..., :))
end

@inline (va::ConstAccessor)(data::AbstractVectorOfSimilarVectors) =
    Fill(va.shape.value, size(data,1))


Base.@propagate_inbounds Base.getindex(data::AbstractVector, va::VariableDataAccessor) = va(data)


Base.@propagate_inbounds Base.view(data::AbstractVector, va::VariableDataAccessor) = va(data)
Base.@propagate_inbounds Base.view(data::AbstractVector{<:AbstractVector}, va::VariableDataAccessor) = va(data)


Base.@propagate_inbounds function Base.view(data::AbstractVector, va::ScalarAccessor)
    view(data, _view_idxs(axes(data, 1), va))
end


Base.@propagate_inbounds function Base.view(
    data::AbstractMatrix, va_row::VariableDataAccessor, va_col::VariableDataAccessor
)
    view(data, _view_idxs(axes(data, 1), va_row), _view_idxs(axes(data, 2), va_col))
end


@inline _varoffset_cumsum_impl(s, x, y, rest...) = (s, _varoffset_cumsum_impl(s+x, y, rest...)...)
@inline _varoffset_cumsum_impl(s,x) = (s,)
@inline _varoffset_cumsum_impl(s) = ()
@inline _varoffset_cumsum(x::Tuple) = _varoffset_cumsum_impl(0, x...)


"""
    VarShapes{N,AC}

Defines the shapes of an ordered set of variables (resp. parameters,
arguments, etc.). This forms the basis of viewing the content of all variables
in a dual way as a `NamedTuple` and as a flattened vectors.

Scalar values have shape `()`, array values have shape `(dim1, dim2, ...)`.

Constructors:

    VarShapes(name1 = shape1, ...)
    VarShapes(varshapes::NamedTuple)

e.g.

    varshapes = VarShapes(
        a = ArrayShape{Real}(2, 3),
        b = ScalarShape{Real}(),
        c = ArrayShape{Real}(4)
    )

Use

    (varshapes::VarShapes)(data::AbstractVector)::NamedTuple

to get correctly named and shaped views into a vector containing the flattened
values of all variables. In return,

    Base.Vector{T}(::UndefInitializer, varshapes::VarShapes)
    Base.Vector(::UndefInitializer, varshapes::VarShapes)

will create a suitable uninitialized vector to hold such flattened data for
a given set of variables. If no type `T` is given, a suitable non-abstract
type will be chosen automatically via `nonabstract_eltype(varshapes)`.

When dealing with multiple vectors of flattened data,

    (varshapes::VarShapes)(
        data::ArrayOfArrays.AbstractVectorOfSimilarVectors
    )::NamedTuple

creates a view of a vector of flattened data vectors as a table with the
variable names as column names and the (possibly array-shaped) variable
value views as entries. In return,

    ArraysOfArrays.VectorOfSimilarVectors{T}(varshapes::VarShapes)
    ArraysOfArrays.VectorOfSimilarVectors(varshapes::VarShapes)

will create a suitable vector (of length zero) of vectors to hold flattened
value data. The result will be a `VectorOfSimilarVectors` wrapped around a
2-dimensional `ElasticArray`. Internally all data is stored in a single
flat `Vector{T}`.

Example:

```julia
varshapes = VarShapes(
    a = ScalarShape{Real}(),
    b = ArrayShape{Real}(2, 3),
    c = ConstValueShape(42)
)
data = VectorOfSimilarVectors{Float64}(varshapes)
resize!(data, 10)
rand!(flatview(data))
table = TypedTables.Table(varshapes(data))
fill!(table.a, 4.2)
all(x -> x == 4.2, view(flatview(data), 1, :))
```
"""
struct VarShapes{names,AT<:(NTuple{N,VariableDataAccessor} where N)}
    _accessors::NamedTuple{names,AT}
    _flatdof::Int

    @inline function VarShapes(varshapes::NamedTuple{names,<:NTuple{N,AbstractValueShape}}) where {names,N}
        labels = keys(varshapes)
        shapes = values(varshapes)
        shapelengths = map(totalndof, shapes)
        offsets = _varoffset_cumsum(shapelengths)
        accessors = map(VariableDataAccessor, shapes, offsets)
        # acclengths = map(x -> x.len, accessors)
        # @assert shapelengths == acclengths
        n_flattened = sum(shapelengths)
        named_accessors = NamedTuple{labels}(accessors)
        new{names,typeof(accessors)}(named_accessors, n_flattened)
    end
end

export VarShapes

@inline VarShapes(;varshapes...) = VarShapes(values(varshapes))


@inline _accessors(x::VarShapes) = getfield(x, :_accessors)
@inline totalndof(x::VarShapes) = getfield(x, :_flatdof)


@inline Base.keys(varshapes::VarShapes) = keys(_accessors(varshapes))

@inline Base.values(varshapes::VarShapes) = values(_accessors(varshapes))

@inline Base.getproperty(varshapes::VarShapes, p::Symbol) = getproperty(_accessors(varshapes), p)

@inline Base.propertynames(varshapes::VarShapes) = propertynames(_accessors(varshapes))

@inline Base.length(varshapes::VarShapes) = length(_accessors(varshapes))

@inline Base.getindex(varshapes::VarShapes, i::Integer) = getindex(_accessors(varshapes), i)

@inline Base.map(f, varshapes::VarShapes) = map(f, _accessors(varshapes))


Base.@propagate_inbounds function (varshapes::VarShapes)(data::AbstractVector)
    accessors = _accessors(varshapes)
    map(va -> va(data), accessors)
end

Base.@propagate_inbounds function (varshapes::VarShapes)(data::AbstractVectorOfSimilarVectors)
    accessors = _accessors(varshapes)
    cols = map(va -> va(data), accessors)
    TypedTables.Table(cols)
end


Base.@pure _multi_promote_type() = Nothing
Base.@pure _multi_promote_type(T::Type) = T
Base.@pure _multi_promote_type(T::Type, U::Type, rest::Type...) = promote_type(T, _multi_promote_type(U, rest...))


Base.@pure nonabstract_eltype(varshapes::VarShapes) =
    _multi_promote_type(map(nonabstract_eltype, values(varshapes))...)


Base.Vector{T}(::UndefInitializer, varshapes::VarShapes) where T =
    Vector{T}(undef, totalndof(varshapes))

Base.Vector(::UndefInitializer, varshapes::VarShapes) =
    Vector{nonabstract_eltype(varshapes)}(undef, totalndof(varshapes))


ArraysOfArrays.VectorOfSimilarVectors{T}(varshapes::VarShapes) where T =
    VectorOfSimilarVectors(ElasticArray{T}(undef, totalndof(varshapes), 0))

ArraysOfArrays.VectorOfSimilarVectors(varshapes::VarShapes) =
    VectorOfSimilarVectors(ElasticArray{nonabstract_eltype(varshapes)}(undef, totalndof(varshapes), 0))
