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
struct ConstValueShape{T, strict} <: AbstractValueShape
    value::T
end

export ConstValueShape

ConstValueShape{T}(x::T) where T = ConstValueShape{T,true}(x)
ConstValueShape(x::T) where T = ConstValueShape{T,true}(x)

Base.:(==)(a::ConstValueShape, b::ConstValueShape) = a.value == b.value
Base.isequal(a::ConstValueShape, b::ConstValueShape) = isequal(a.value, b.value)
Base.isapprox(a::ConstValueShape, b::ConstValueShape; kwargs...) = isapprox(a.value, b.value; kwargs...)
Base.hash(x::ConstValueShape, h::UInt) = hash(x.value, hash(:ConstValueShape, hash(:ValueShapes, h)))


@inline Base.size(shape::ConstValueShape) = size(shape.value)
@inline Base.length(shape::ConstValueShape) = length(shape.value)


@inline Base.:(<=)(a::ConstValueShape{T}, b::ConstValueShape{U}) where {T,U} = T<:U && a.value ≈ b.value


@inline default_unshaped_eltype(shape::ConstValueShape) = Int32

@inline shaped_type(shape::ConstValueShape, ::Type{T}) where {T<:Real} = typeof(shape.value)


@inline totalndof(::ConstValueShape) = 0


# ToDo/Decision: Return a copy instead?
(shape::ConstValueShape)(::UndefInitializer) = shape.value


function unshaped(x::Any, shape::ConstValueShape)
    x == shape.value || throw(ArgumentError("Given value does not match value of ConstValueShape"))
    Float32[]
end

@inline _valshapeoftype(::Type{Nothing}) = ConstValueShape(nothing)



"""
    const_zero_shape(shape::ConstValueShape)

Get the equivalent of a constant zero shape for shape `shape` that will
only allow zero values to be set via an accessor.
"""
const_zero_shape(shape::ConstValueShape) = ConstValueShape(const_zero(shape.value))


"""
    nonstrict_const_zero_shape(shape::ConstValueShape)

Get the equivalent of a constant zero shape for shape `shape` that will
ignore any attempt to set a value via an accessor.

Useful as a gradient/tangent varshape of constants, as they can ignore
attempts to set non-zero values.
"""
function nonstrict_const_zero_shape(shape::ConstValueShape)
    x = const_zero(shape.value)
    ConstValueShape{typeof(x),false}(x)
end


replace_const_shapes(f::Function, shape::ConstValueShape) = f(shape)



const ConstAccessor{T,strict} = ValueAccessor{ConstValueShape{T,strict}}


@inline vs_getindex(data::AbstractVector{<:Real}, va::ConstAccessor) = va.shape.value

@inline vs_unsafe_view(::AbstractVector{<:Real}, va::ConstAccessor) = va.shape.value

# Zygote has a generic `@adjoint getindex(x::AbstractArray, inds...)` and same for view that
# will result in overwriting va.shape.value with dy without these custom adjoints:
ZygoteRules.@adjoint function getindex(x::AbstractVector{<:Real}, va::ConstAccessor)
    getindex(x, va), dy -> nothing, nothing
end
ZygoteRules.@adjoint function view(x::AbstractVector{<:Real}, va::ConstAccessor)
    view(x, va), dy -> nothing, nothing
end


function vs_setindex!(data::AbstractVector{<:Real}, v, va::ConstAccessor{T,true}) where T
    v == va.shape.value || throw(ArgumentError("Cannot set constant value to a different value"))
    data
end

function vs_setindex!(data::AbstractVector{<:Real}, v, va::ConstAccessor{T,false}) where T
    data
end


@inline _bcasted_view(data::AbstractArrayOfSimilarVectors{<:Real,N}, va::ConstAccessor) where N =
    Fill(va.shape.value, size(data)...)

Base.Broadcast.broadcasted(::typeof(getindex), A::AbstractArrayOfSimilarVectors{<:Real,N}, acc::Ref{<:ConstAccessor}) where N =
    _bcasted_view(A, acc[])

Base.Broadcast.broadcasted(::typeof(view), A::AbstractArrayOfSimilarVectors{<:Real,N}, acc::Ref{<:ConstAccessor}) where N =
    _bcasted_view(A, acc[])
