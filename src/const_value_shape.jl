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

@inline Base.eltype(shape::ConstValueShape) = Int

@inline totalndof(::ConstValueShape) = 0


# ToDo/Decision: Return a copy instead?
(shape::ConstValueShape)(::UndefInitializer) = shape.value


@static if VERSION < v"1.3"
    # Workaround for Julia issue #14919
    @inline (shape::ConstValueShape)(data::AbstractVector{<:Real}) =
        _apply_shape_to_data(shape, data)
end



const ConstAccessor = ValueAccessor{ConstValueShape{T}} where {T}


@inline Base.getindex(data::AbstractVector{<:Real}, va::ConstAccessor) = view(data, va)

@inline Base.view(::AbstractVector, va::ConstAccessor) = va.shape.value


@inline _bcasted_getindex(data::AbstractVectorOfSimilarVectors{<:Real}, va::ConstAccessor) =
    Fill(va.shape.value, size(data,1))

Base.copy(instance::VSBroadcasted2{typeof(getindex),AbstractVectorOfSimilarVectors{<:Real},Ref{<:ConstAccessor}}) =
    _bcasted_getindex(instance.args[1], instance.args[2][])    
