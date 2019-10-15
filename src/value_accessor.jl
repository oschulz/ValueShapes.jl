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




Base.@propagate_inbounds function Base.view(
    data::AbstractMatrix, va_row::ValueAccessor, va_col::ValueAccessor
)
    view(data, view_idxs(axes(data, 1), va_row), view_idxs(axes(data, 2), va_col))
end
