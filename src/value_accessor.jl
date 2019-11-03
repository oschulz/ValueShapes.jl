# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    ValueAccessor{S<:AbstractValueShape}

A value accessor provides a means to access a value with a given shape
stored in a flat real-valued data vector with a given offset position.

Constructor:

    ValueAccessor{S}(shape::S, offset::Int)

The offset is relative to the first index of a flat data array, so if
the value is stored at the beginning of the array, the offset will be zero.

An `ValueAccessor` can be used to index into a given flat data array.

Example:

```julia
acc = ValueAccessor(ArrayShape{Real}(2,3), 2)
valshape(acc) == ArrayShape{Real,2}((2, 3))
data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
data[acc] == [3 5 7; 4 6 8]

Note: Subtypes of [`AbstractValueShape`](@ref) should specialize
[`ValueShapes.vs_getindex`](@ref), [`ValueShapes.vs_unsafe_view`](@ref) and
[`ValueShapes.vs_setindex!`](@ref) for their `ValueAccessor{...}`.
Specializing `Base.getindex`, `Base.view`, `Base.unsafe_view` or
`Base.setindex!` directly may result in method ambiguities with custom array
tapes that specialize these functions in a very generic fashion.
```
"""
struct ValueAccessor{S<:AbstractValueShape}
    shape::S
    offset::Int
    len::Int

    ValueAccessor{S}(shape::S, offset::Int) where {S<:AbstractValueShape} =
        new{S}(shape, offset, totalndof(shape))
end

export ValueAccessor

ValueAccessor(shape::S, offset::Int) where {S<:AbstractValueShape} = ValueAccessor{S}(shape, offset)


# Value accessors behave as scalars under broadcasting:
@inline Base.Broadcast.broadcastable(shape::ValueAccessor) = Ref(shape)


nonabstract_eltype(va::ValueAccessor) = nonabstract_eltype(va.shape)


Base.size(va::ValueAccessor) = size(va.shape)
Base.length(va::ValueAccessor) = length(va.shape)

Base.to_indices(A::AbstractArray{T,N}, I::NTuple{N,ValueAccessor}) where {T<:Real,N} = I

Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::ValueAccessor) =
    checkindex(Bool, inds, view_idxs(inds, i))


valshape(va::ValueAccessor) = va.shape

# Would this be useful?
# AbstractValueShape(va::ValueAccessor) = valshape(va)
# Base.convert(::Type{AbstractValueShape}, va::ValueAccessor) = AbstractValueShape(va)


function view_range end

@inline function view_range(idxs::AbstractUnitRange{<:Integer}, va::ValueAccessor)
    from = first(idxs) + va.offset
    to = from + va.len - 1
    from:to
end


function view_idxs end

@inline view_idxs(idxs::AbstractUnitRange{<:Integer}, va::ValueAccessor) = view_range(idxs, va)


"""
    ValueShapes.vs_getindex(data::AbstractArray{<:Real}, idxs::ValueAccessor...)

Specialize `ValueShapes.vs_getindex` instead of `Base.getindex` for
[`ValueShapes.ValueAccessor`](@ref)s, to avoid methods ambiguities with
with certain custom array types.
"""
function vs_getindex end

Base.@propagate_inbounds function vs_getindex(
    data::AbstractMatrix{<:Real}, va_row::ValueAccessor, va_col::ValueAccessor
)
    getindex(data, view_idxs(axes(data, 1), va_row), view_idxs(axes(data, 2), va_col))
end

Base.@propagate_inbounds Base._getindex(::IndexStyle, data::AbstractVector{<:Real}, idx::ValueAccessor) =
    vs_getindex(data, idx)

Base.@propagate_inbounds Base._getindex(::IndexStyle, data::AbstractArray{<:Real,N}, idxs::Vararg{ValueAccessor,N}) where N =
    vs_getindex(data, idxs...)


"""
    ValueShapes.vs_unsafe_view(data::AbstractArray{<:Real}, idxs::ValueAccessor...)

Specialize `ValueShapes.vs_unsafe_view` instead of `Base.view` or
`Base.unsafe_view` for [`ValueShapes.ValueAccessor`](@ref)s, to avoid methods
ambiguities with with certain custom array types.
"""
function vs_unsafe_view end

Base.@propagate_inbounds function vs_unsafe_view(
    data::AbstractMatrix{<:Real}, va_row::ValueAccessor, va_col::ValueAccessor
)
    Base.unsafe_view(data, view_idxs(axes(data, 1), va_row), view_idxs(axes(data, 2), va_col))
end

Base.@propagate_inbounds Base.unsafe_view(data::AbstractVector{<:Real}, idx::ValueAccessor) =
    vs_unsafe_view(data, idx)

Base.@propagate_inbounds Base.unsafe_view(data::AbstractArray{<:Real,N}, idxs::Vararg{ValueAccessor,N}) where N =
    vs_unsafe_view(data, idxs...)


"""
    ValueShapes.vs_setindex!(data::AbstractArray{<:Real}, v, idxs::ValueAccessor...)

Specialize `ValueShapes.vs_setindex!` instead of `Base.setindex!` or for
[`ValueShapes.ValueAccessor`](@ref)s, to avoid methods ambiguities with with
certain custom array types.
"""
function vs_setindex! end

Base.@propagate_inbounds function vs_setindex!(
    data::AbstractMatrix{<:Real}, v, va_row::ValueAccessor, va_col::ValueAccessor
)
    setindex!(data, v, view_idxs(axes(data, 1), va_row), view_idxs(axes(data, 2), va_col))
end

Base.@propagate_inbounds Base._setindex!(::IndexStyle, data::AbstractVector{<:Real}, v, idx::ValueAccessor) =
    vs_setindex!(data, v, idx)

Base.@propagate_inbounds Base._setindex!(::IndexStyle, data::AbstractArray{<:Real,N}, v, idxs::Vararg{ValueAccessor,N}) where N =
    vs_setindex!(data, v, idxs...)
