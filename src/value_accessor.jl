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
```

Note: Subtypes of [`AbstractValueShape`](@ref) should specialize
[`ValueShapes.vs_getindex`](@ref), [`ValueShapes.vs_unsafe_view`](@ref) and
[`ValueShapes.vs_setindex!`](@ref) for their `ValueAccessor{...}`.
Specializing `Base.getindex`, `Base.view`, `Base.unsafe_view` or
`Base.setindex!` directly may result in method ambiguities with custom array
tapes that specialize these functions in a very generic fashion.
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


# Reserve broadcasting semantics for value accessors:
@inline Base.Broadcast.broadcastable(va::ValueAccessor) =
    throw(ArgumentError("broadcasting over `ValueAccessor`s is reserved"))


default_unshaped_eltype(va::ValueAccessor) = default_unshaped_eltype(va.shape)


Base.size(va::ValueAccessor) = size(va.shape)
Base.length(va::ValueAccessor) = length(va.shape)

# Can't use `idxs::Vararg{ValueAccessor,N}`, would cause ambiguities with
# Base for N == 0.
Base.to_indices(A::AbstractArray{T,1}, I::Tuple{ValueAccessor}) where {T<:Real} = I
Base.to_indices(A::AbstractArray{T,2}, I::Tuple{ValueAccessor,ValueAccessor}) where {T<:Real} = I
Base.to_indices(A::AbstractArray{T,3}, I::Tuple{ValueAccessor,ValueAccessor,ValueAccessor}) where {T<:Real} = I

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

# Can't use `idxs::Vararg{ValueAccessor,N}`, would cause ambiguities with
# Base for N == 0.
Base.@propagate_inbounds Base._getindex(::IndexStyle, data::AbstractArray{<:Real,1}, idx1::ValueAccessor) =
    vs_getindex(data, idx1)
Base.@propagate_inbounds Base._getindex(::IndexStyle, data::AbstractArray{<:Real,2}, idx1::ValueAccessor, idx2::ValueAccessor) =
    vs_getindex(data, idx1, idx2)
Base.@propagate_inbounds Base._getindex(::IndexStyle, data::AbstractArray{<:Real,3}, idx1::ValueAccessor, idx2::ValueAccessor, idx3::ValueAccessor) =
    vs_getindex(data, idx1, idx2, idx3)


"""
    ValueShapes.vs_unsafe_view(data::AbstractArray{<:Real}, idxs::ValueAccessor...)

Specialize `ValueShapes.vs_unsafe_view` instead of `Base.view` or
`Base.unsafe_view` for [`ValueShapes.ValueAccessor`](@ref)s, to avoid methods
ambiguities with with certain custom array types.
"""
function vs_unsafe_view end

# Can't use `idxs::Vararg{ValueAccessor,N}`, would cause ambiguities with
# Base for N == 0.
Base.@propagate_inbounds Base.unsafe_view(data::AbstractArray{<:Real,1}, idx1::ValueAccessor) =
    vs_unsafe_view(data, idx1)
Base.@propagate_inbounds Base.unsafe_view(data::AbstractArray{<:Real,2}, idx1::ValueAccessor, idx2::ValueAccessor) =
    vs_unsafe_view(data, idx1, idx2)
Base.@propagate_inbounds Base.unsafe_view(data::AbstractArray{<:Real,3}, idx1::ValueAccessor, idx2::ValueAccessor, idx3::ValueAccessor) =
    vs_unsafe_view(data, idx1, idx2, idx3)


"""
    ValueShapes.vs_setindex!(data::AbstractArray{<:Real}, v, idxs::ValueAccessor...)

Specialize `ValueShapes.vs_setindex!` instead of `Base.setindex!` or for
[`ValueShapes.ValueAccessor`](@ref)s, to avoid methods ambiguities with with
certain custom array types.
"""
function vs_setindex! end

# Can't use `idxs::Vararg{ValueAccessor,N}`, would cause ambiguities with
# Base for N == 0.
Base.@propagate_inbounds Base._setindex!(::IndexStyle, data::AbstractArray{<:Real,1}, v, idx1::ValueAccessor) =
    vs_setindex!(data, v, idx1)
Base.@propagate_inbounds Base._setindex!(::IndexStyle, data::AbstractArray{<:Real,2}, v, idx1::ValueAccessor, idx2::ValueAccessor) =
    vs_setindex!(data, v, idx1, idx2)
Base.@propagate_inbounds Base._setindex!(::IndexStyle, data::AbstractArray{<:Real,3}, v, idx1::ValueAccessor, idx2::ValueAccessor, idx3::ValueAccessor) =
    vs_setindex!(data, v, idx1, idx2, idx3)
