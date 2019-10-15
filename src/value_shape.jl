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

    Base.Vector{T}(undef, shape::AbstractValueShape)
    Base.Vector(undef, shape::AbstractValueShape)

will create a suitable uninitialized vector of the right length to hold such
flat data for the given shape. If no type `T` is given, a suitable data
type will be chosen automatically.

When dealing with multiple vectors of flattened data, use
    
    shape.(data::ArrayOfArrays.AbstractVectorOfSimilarVectors)

ValueShapes supports this via specialized broadcasting.

In return,

    ArraysOfArrays.VectorOfSimilarVectors{T}(shape::AbstractValueShape)
    ArraysOfArrays.VectorOfSimilarVectors(shape::AbstractValueShape)

will create a suitable vector (of length zero) of vectors that can hold
flattened data for the given shape. The result will be a
`VectorOfSimilarVectors` wrapped around a 2-dimensional `ElasticArray`.
This way, all data is stored in a single contiguous chunk of memory.
"""
abstract type AbstractValueShape end

export AbstractValueShape


# Value shapes behave as scalars under broadcasting:
@inline Base.Broadcast.broadcastable(shape::AbstractValueShape) = Ref(shape)


function _valshapeoftype end


"""
    valshape(x)::AbstractValueShape
    valshape(acc::ValueAccessor)::AbstractValueShape
    valshape(d::Distributions.Distribution)::AbstractValueShape

Get the value shape of an arbitrary value, resp. the shape a `ValueAccessor`
is based on, or the shape of the variates for a `Distribution`.
"""
function valshape end
export valshape

@inline valshape(x::T) where T = _valshapeoftype(T)


"""
    totalndof(shape::AbstractValueShape)

Get the total number of degrees of freedom of values of the given shape.

Equivalent to the length of a vector that would result from flattening the
data into a sequence of real numbers, excluding any constant values.
"""
function totalndof end
export totalndof


@inline nonabstract_eltype(shape::AbstractValueShape) = default_datatype(eltype(shape))


function _checkcompat(shape::AbstractValueShape, data::AbstractVector{<:Real})
    n_shape = totalndof(shape)
    n_data = length(eachindex(data))
     if n_shape != length(eachindex(data))
        throw(ArgumentError("Data vector of length $(n_data) incompatible with value shape with $(n_shape) degrees of freedom"))
    end
    nothing
end


@static if VERSION >= v"1.3"
    @inline function (shape::AbstractValueShape)(data::AbstractVector{<:Real})
        @boundscheck _checkcompat(shape, data)
        data[ValueAccessor(shape, 0)]
    end
else
    # Workaround for Julia issue #14919
    @inline function _apply_shape_to_data(shape::AbstractValueShape, data::AbstractVector{<:Real})
        @boundscheck _checkcompat(shape, data)
        data[ValueAccessor(shape, 0)]
    end
end


Base.Vector{T}(::UndefInitializer, shape::AbstractValueShape) where T =
    Vector{T}(undef, totalndof(shape))

Base.Vector(::UndefInitializer, shape::AbstractValueShape) =
    Vector{nonabstract_eltype(shape)}(undef, totalndof(shape))


ArraysOfArrays.VectorOfSimilarVectors{T}(shape::AbstractValueShape) where T =
    VectorOfSimilarVectors(ElasticArray{T}(undef, totalndof(shape), 0))

ArraysOfArrays.VectorOfSimilarVectors(shape::AbstractValueShape) =
    VectorOfSimilarVectors(ElasticArray{nonabstract_eltype(shape)}(undef, totalndof(shape), 0))


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
    broadcast(getindex, instance.args[1], ValueAccessor(instance.f, 0))    
