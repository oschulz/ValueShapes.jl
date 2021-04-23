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

`AbstractValueShape`s can be compared with `<=` and `>=`, with semantics that
are similar to compare type with `<:` and `>:`:

```julia
a::AbstractValueShape <= b::AbstractValueShape == true
```

implies that values of shape `a` are can be used in contexts that expect
values of shape `b`. E.g.:

```julia
(ArrayShape{Float64}(4,5) <= ArrayShape{Real}(4,5)) == true
(ArrayShape{Float64}(4,5) <= ArrayShape{Integer}(4,5)) == false
(ArrayShape{Float64}(2,2) <= ArrayShape{Float64}(3,3)) == false
(ScalarShape{Real}() >= ScalarShape{Int}()) == true
```
"""
abstract type AbstractValueShape end

export AbstractValueShape


import Base.>=
@inline >=(a::AbstractValueShape, b::AbstractValueShape) = b <= a


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

@inline elshape(A::AbstractArray{<:AbstractArray}) = ArrayShape{eltype(eltype(A))}(innersize(A)...)


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
    unshaped(x, shape::AbstractValueShape)::AbstractVector{<:Real}

Retrieve the unshaped underlying data of x, assuming x is a structured view
(based on some [`AbstractValueShape`](@ref)) of a flat/unstructured
real-valued data vector.

If `shape` is given, ensures that the shape of `x` is compatible with it.
Specifying a shape may be necessary if the correct shape of `x` cannot be
inferred from `x`, e.g. because `x` is assumed to have fewer degrees of
freedom (because of constant components) than would be inferred from
the plain value of `x`.

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

unshaped(x::Real) = Fill(x, 1)
unshaped(x::AbstractArray{<:Real,0}) = view(x, firstindex(x):firstindex(x))
unshaped(x::SubArray{<:Real,0}) = view(parent(x), x.indices[1]:x.indices[1])
unshaped(x::AbstractArray{<:Real,1}) = x
unshaped(x::Base.ReshapedArray{T,N,<:AbstractArray{T,1}}) where {T<:Real,N} = parent(x)


"""
    stripscalar(x)

Dereference value `x`.

If x is a scalar-like object, like a 0-dimensional array or a `Ref`,
`stripscalar` returns it's inner value. Otherwise, `x` is returned unchanged.

Useful to strip shaped scalar-like views of their 0-dim array semantics
(if present), but leave array-like views unchanged.

Example:

```julia
data = [1, 2, 3]
shape1 = NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Real}(2))
x1 = shape1(data)
@assert x1 isa AbstractArray{<:NamedTuple,0}
@assert stripscalar(x) isa NamedTuple

shape2 = ArrayShape{Real}(3)
x2 = shape2(data)
@assert x2 isa AbstractArray{Int,1}
@assert ref(x2) isa AbstractArray{Int,1}
```
"""
function stripscalar end
export stripscalar

stripscalar(x::Any) = x
stripscalar(x::Ref) = x[]
stripscalar(x::AbstractArray{T,0}) where T = x[]


function _checkcompat(shape::AbstractValueShape, data::AbstractVector{<:Real})
    n_shape = totalndof(shape)
    n_data = length(eachindex(data))
    if n_shape != n_data
        throw(ArgumentError("Data vector of length $(n_data) incompatible with value shape with $(n_shape) degrees of freedom"))
    end
    nothing
end


function _checkcompat_inner(shape::AbstractValueShape, data::AbstractArray{<:AbstractVector{<:Real}})
    n_shape = totalndof(shape)
    n_data = prod(innersize(data))
    if n_shape != n_data
        throw(ArgumentError("Data vector of length $(n_data) incompatible with value shape with $(n_shape) degrees of freedom"))
    end
    nothing
end


@inline function _apply_shape_to_data(shape::AbstractValueShape, data::AbstractVector{<:Real})
    @boundscheck _checkcompat(shape, data)
    view(data, ValueAccessor(shape, 0))
end


@inline function (shape::AbstractValueShape)(data::AbstractVector{<:Real})
    _apply_shape_to_data(shape, data)
end


Base.Vector{T}(::UndefInitializer, shape::AbstractValueShape) where {T <: Real} =
    Vector{T}(undef, totalndof(shape))

Base.Vector{<:Real}(::UndefInitializer, shape::AbstractValueShape) =
    Vector{default_unshaped_eltype(shape)}(undef, shape)


ArraysOfArrays.VectorOfSimilarVectors{T}(shape::AbstractValueShape) where {T<:Real} =
    VectorOfSimilarVectors(ElasticArray{T}(undef, totalndof(shape), 0))


const VSBroadcasted1{N,F,T} = Base.Broadcast.Broadcasted{
    <:Base.Broadcast.AbstractArrayStyle{N},
    <:Any,
    F,
    <:Tuple{T}
}


const VSBroadcasted2{N,F,T1,T2} = Base.Broadcast.Broadcasted{
    <:Base.Broadcast.AbstractArrayStyle{N},
    <:Any,
    F,
    <:Tuple{T1,T2}
}


# Specialize (::AbstractValueShape).(::AbstractVector{<:AbstractVector{<:Real}}):
Base.copy(instance::VSBroadcasted1{N,<:AbstractValueShape,AbstractArray{<:AbstractVector{<:Real},N}}) where N =
    broadcast(view, instance.args[1], Ref(ValueAccessor(instance.f, 0)))


# Specialize unshaped for real vectors (semantically vectors of scalar-shaped values)
function Base.copy(instance::VSBroadcasted1{1,typeof(unshaped),AbstractVector{<:Real}})
    x = instance.args[1]
    nestedview(reshape(view(x, :), 1, length(eachindex(x))))
end

# Specialize unshaped for real vectors that are array slices:
const _MatrixSliceFirstDim{T} = SubArray{T,1,<:AbstractArray{T,2},<:Tuple{Int,AbstractArray{Int}}}
function Base.copy(instance::VSBroadcasted1{1,typeof(unshaped),<:_MatrixSliceFirstDim{<:Real}})
    instance
    x = instance.args[1]
    nestedview(view(parent(x), x.indices[1]:x.indices[1], x.indices[2]))
end


function _zerodim_array(x::T) where T
    A = Array{T,0}(undef)
    A[] = x
end


"""
    const_zero(x::Any)

Get the equivalent of a constant zero for values the same type as .
"""
function const_zero end

const_zero(x::Number) = zero(x)
const_zero(A::AbstractArray{T}) where T <: Number = Fill(zero(T), size(A)...)


"""
    replace_const_shapes(f::Function, shape::AbstractValueShape)

If `shape` is a, or contains, [`ConstValueShape`](@ref) shape(s), recursively
replace it/them with the result of `f(s::Shape)`.
"""
function replace_const_shapes end
export replace_const_shapes


"""
    gradient_shape(argshape::AbstractValueShape)

Return the value shape of the gradient of functions that take values of
shape `argshape` as an input.
"""
function gradient_shape end

gradient_shape(vs::AbstractValueShape) = replace_const_shapes(const_zero_shape, vs)
export gradient_shape
