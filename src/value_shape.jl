# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    ValueShapes.default_datatype(T::Type)

Return a default specific type U that is more specific than T, with U <: T.

e.g.

    ValueShapes.default_datatype(Real) == Float64
    ValueShapes.default_datatype(Complex) == Complex{Float64}
"""
function default_datatype end

@inline default_datatype(::Type{>:Int}) = Int
@inline default_datatype(::Type{>:Float64}) = Float64
@inline default_datatype(::Type{>:Real}) = Float64
@inline default_datatype(::Type{>:Complex{Float64}}) = Complex{Float64}
@inline default_datatype(T::Type) = T



"""
    abstract type AbstractValueShape

An `AbstractValueShape` combines type and size information.

Subtypes are defined for shapes of scalars (see [`ScalarShape`](@ref)),
arrays (see [`ArrayShape`](@ref)), constant values
(see [`ConstValueShape`](@ref)) and `NamedTuple`s (see
[`NamedTupleShape`](@ref)).

Subtypes of `AbstractValueShape` must support `eltype`, `size` and
[`totalndof`](@ref).

Value shapes can be used as constructors to generate values of the given
shape with undefined content. If the element type of the shape is an abstract
or union type, a suitable concrete type will be chosen automatically, if
possible (see [`ValueShapes.default_datatype`](@ref)):

```julia
shape = ArrayShape{Real}(2,3)
A = shape(undef)
typeof(A) == Array{Float64,2}
size(A) == (2, 3)
valshape(A) == ArrayShape{Float64}(2,3)
```

Use

    (shape::AbstractValueShape)(data::AbstractVector{<:Real})::eltype(shape)

to view a flat vector of anonymous real values
as a value of the given shape:

```julia
data = [1, 2, 3, 4, 5, 6]
shape(data) == [1 3 5; 2 4 6]
```

In return,

    Base.Vector{<:Real}(undef, shape::AbstractValueShape)

will create a suitable uninitialized vector of the right length to hold such
flat data for the given shape. If no type `T` is given, a suitable data
type will be chosen automatically.

When dealing with multiple vectors of flattened data, use
    
    shape.(data::ArraysOfArrays.AbstractVectorOfSimilarVectors)

ValueShapes supports this via specialized broadcasting.

In return,

    ArraysOfArrays.VectorOfSimilarVectors{<:Real}(shape::AbstractValueShape)

will create a suitable vector (of length zero) of vectors that can hold
flattened data for the given shape. The result will be a
`VectorOfSimilarVectors` wrapped around a 2-dimensional `ElasticArray`.
This way, all data is stored in a single contiguous chunk of memory.
"""
abstract type AbstractValueShape end

export AbstractValueShape


# Reserve broadcasting semantics for value shapes:
@inline Base.Broadcast.broadcastable(shape::AbstractValueShape) =
    throw(ArgumentError("broadcasting over `AbstractValueShape`s is reserved"))


function _valshapeoftype end


"""
    ValueShapes.default_unshaped_eltype(shape::AbstractValueShape)

Returns the default real array element type to use for unshaped
representations of data with shape `shape`.

Subtypes of `AbstractValueShape` must implemenent
`ValueShapes.default_unshaped_eltype`.
"""
function default_unshaped_eltype end


"""
    ValueShapes.shaped_type(shape::AbstractValueShape, ::Type{T}) where {T<:Real}
    ValueShapes.shaped_type(shape::AbstractValueShape)

Returns the type the will result from reshaping a real-valued vector (of
element type `T`, if specified) with `shape`.

Subtypes of `AbstractValueShape` must implement

    ValueShapes.shaped_type(shape::AbstractValueShape, ::Type{T}) where {T<:Real}
"""
function shaped_type end

shaped_type(shape::AbstractValueShape) = shaped_type(shape, default_unshaped_eltype(shape))


"""
    valshape(x)::AbstractValueShape
    valshape(acc::ValueAccessor)::AbstractValueShape

Get the value shape of an arbitrary value, resp. the shape a `ValueAccessor`
is based on, or the shape of the variates for a `Distribution`.
"""
function valshape end
export valshape

@inline valshape(x::T) where T = _valshapeoftype(T)


"""
    elshape(x)::AbstractValueShape

Get the shape of the elements of x
"""
function elshape end
export elshape

@inline elshape(x::T) where T = _valshapeoftype(eltype(T))

@inline elshape(A::AbstractArray{T,N}) where {T,N} = ArrayShape{T}{innersize(A)}


"""
    totalndof(shape::AbstractValueShape)

Get the total number of degrees of freedom of values of the given shape.

Equivalent to the length of a vector that would result from flattening the
data into a sequence of real numbers, excluding any constant values.
"""
function totalndof end
export totalndof


"""
    unshaped(x)::AbstractVector{<:Real}

Retrieve the unshaped underlying data of x, assuming x is a structured view
(based on some [`AbstractValueShape`](@ref)) of a flat/unstructured
real-valued data vector.

Example:

```julia
shape = NamedTupleShape(
    a = ScalarShape{Real}(),
    b = ArrayShape{Real}(2, 3)
)
data = [1, 2, 3, 4, 5, 6, 7]
x = shape(data)
@assert unshaped(x) == data
@assert unshaped(x.a) == view(data, 1:1)
@assert unshaped(x.b) == view(data, 2:7)
```
"""
function unshaped end
export unshaped

unshaped(x::SubArray{T,0}) where T = view(parent(x), x.indices[1]:x.indices[1])
unshaped(x::SubArray{T,1}) where T = x
unshaped(x::Base.ReshapedArray{T,N,<:AbstractArray{T,1}}) where {T,N} = parent(x)


function _checkcompat(shape::AbstractValueShape, data::AbstractVector{<:Real})
    n_shape = totalndof(shape)
    n_data = length(eachindex(data))
    if n_shape != length(eachindex(data))
        throw(ArgumentError("Data vector of length $(n_data) incompatible with value shape with $(n_shape) degrees of freedom"))
    end
    nothing
end


@inline function _apply_shape_to_data(shape::AbstractValueShape, data::AbstractVector{<:Real})
    @boundscheck _checkcompat(shape, data)
    view(data, ValueAccessor(shape, 0))
end

@static if VERSION >= v"1.3"
    @inline function (shape::AbstractValueShape)(data::AbstractVector{<:Real})
        _apply_shape_to_data(shape, data)
    end
else
    # With Julia < v1.3, need to define (shape::SomeShape)(...) explicitly for
    # each shape type, see source code for the respective shapes.
end


Base.Vector{T}(::UndefInitializer, shape::AbstractValueShape) where {T <: Real} =
    Vector{T}(undef, totalndof(shape))


ArraysOfArrays.VectorOfSimilarVectors{T}(shape::AbstractValueShape) where {T<:Real} =
    VectorOfSimilarVectors(ElasticArray{T}(undef, totalndof(shape), 0))


const VSBroadcasted1{F,T} = Base.Broadcast.Broadcasted{
    <:Base.Broadcast.AbstractArrayStyle{1},
    <:Any,
    F,
    <:Tuple{T}
}


const VSBroadcasted2{F,T1,T2} = Base.Broadcast.Broadcasted{
    <:Base.Broadcast.AbstractArrayStyle{1},
    <:Any,
    F,
    <:Tuple{T1,T2}
}


# Specialize (::AbstractValueShape).(::AbstractVector{<:AbstractVector}):
Base.copy(instance::VSBroadcasted1{<:AbstractValueShape,AbstractVector{<:AbstractVector}}) =
    broadcast(view, instance.args[1], Ref(ValueAccessor(instance.f, 0)))


function _zerodim_array(x::T) where T
    A = Array{T,0}(undef)
    A[0] = x
end
