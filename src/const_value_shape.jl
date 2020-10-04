# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    ConstValueShape{T} <: AbstractValueShape

A `ConstValueShape` describes the shape of constant values of type `T`.

Constructor:

    ConstValueShape(value)

`value` may be of arbitrary type, e.g. a constant scalar value or array:

    ConstValueShape(4.2)
    ConstValueShape([11 21; 12 22])

Shapes of constant values have zero degrees of freedom (see
[`totalndof`](@ref)).

See also the documentation of [`AbstractValueShape`](@ref).
"""
struct ConstValueShape{T} <: AbstractValueShape
    value::T
end

export ConstValueShape


@inline Base.size(shape::ConstValueShape) = size(shape.value)
@inline Base.length(shape::ConstValueShape) = length(shape.value)


import Base.<=
@inline <=(a::ConstValueShape{T}, b::ConstValueShape{U}) where {T,U} = T<:U && a.value ≈ b.value


@inline default_unshaped_eltype(shape::ConstValueShape) = Int32

@inline shaped_type(shape::ConstValueShape, ::Type{T}) where {T<:Real} = typeof(shape.value)


@inline totalndof(::ConstValueShape) = 0


# ToDo/Decision: Return a copy instead?
(shape::ConstValueShape)(::UndefInitializer) = shape.value


function unshaped(x::Any, shape::ConstValueShape)
    x == shape.value || throw(ArgumentError("Given value does not match value of ConstValueShape"))
    Float32[]
end


"""
    const_zero_shape(shape::ConstValueShape)

Get the equivalent of a constant zero shape for shape `shape`.
"""
const_zero_shape(shape::ConstValueShape) = ConstValueShape(const_zero(shape.value))


replace_const_shapes(f::Function, shape::ConstValueShape) = f(shape)


@static if VERSION < v"1.3"
    # Workaround for Julia issue #14919
    @inline (shape::ConstValueShape)(data::AbstractVector{<:Real}) =
        _apply_shape_to_data(shape, data)
end



const ConstAccessor = ValueAccessor{ConstValueShape{T}} where {T}


@inline vs_getindex(data::AbstractVector{<:Real}, va::ConstAccessor) = va.shape.value

@inline vs_unsafe_view(::AbstractVector, va::ConstAccessor) = va.shape.value


function vs_setindex!(data::AbstractVector{<:Real}, v, va::ConstAccessor)
    v == va.shape.value || throw(ArgumentError("Cannot set constant value to a different value"))
    data
end


@inline _bcasted_view(data::AbstractArrayOfSimilarVectors{<:Real,N}, va::ConstAccessor) where N =
    Fill(va.shape.value, size(data)...)

Base.copy(instance::VSBroadcasted2{N,typeof(getindex),AbstractArrayOfSimilarVectors{<:Real,N},Ref{<:ConstAccessor}}) where N =
    _bcasted_view(instance.args[1], instance.args[2][])

Base.copy(instance::VSBroadcasted2{N,typeof(view),AbstractArrayOfSimilarVectors{<:Real,N},Ref{<:ConstAccessor}}) where N =
    _bcasted_view(instance.args[1], instance.args[2][])
