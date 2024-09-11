# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    realnumtype(T::Type)

Return the underlying numerical type of T that's a subtype of `Real`.

Uses type promotion among underlying `Real` type in `T`.

e.g.

```julia

A = fill(fill(rand(Float32, 5), 10), 5)
realnumtype(typeof(A)) == Float32
```
"""
function realnumtype end
export realnumtype

realnumtype(::Type{T}) where T = throw(ArgumentError("Can't derive numeric type for type $T"))

realnumtype(::Type{T}) where {T<:Real} = T
realnumtype(::Type{<:Complex{T}}) where {T<:Real} = T
realnumtype(::Type{<:Enum{T}}) where {T<:Real} = T
realnumtype(::Type{<:AbstractArray{T}}) where {T} = realnumtype(T)
realnumtype(::Type{<:NamedTuple{names,T}}) where {names,T} = realnumtype(T)

realnumtype(::Type{NTuple{N,T}}) where {N,T} = realnumtype(T)

@generated function realnumtype(::Type{T}) where {T<:Tuple}
    :(promote_type(map(realnumtype, $((T.parameters...,)))...))
end

# Use a fake numtype for non-numerical types that may be used to express
# default and missing values, string/symbol options, etc.:
realnumtype(::Type{Nothing}) = Bool
realnumtype(::Type{Missing}) = Bool
realnumtype(::Type{Tuple{}}) = Bool
realnumtype(::Type{Symbol}) = Bool
realnumtype(::Type{<:AbstractString}) = Bool


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


@inline Base.:(>=)(a::AbstractValueShape, b::AbstractValueShape) = b <= a

vs_cmp_pullback(ΔΩ) = (NoTangent(), NoTangent(), NoTangent())
ChainRulesCore.rrule(::typeof(Base.:(==)), a::AbstractValueShape, b::AbstractValueShape) = (a == b, vs_cmp_pullback)
ChainRulesCore.rrule(::typeof(Base.:(<=)), a::AbstractValueShape, b::AbstractValueShape) = (a <= b, vs_cmp_pullback)
ChainRulesCore.rrule(::typeof(Base.:(>=)), a::AbstractValueShape, b::AbstractValueShape) = (a >= b, vs_cmp_pullback)


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

# Support for missing varshapes:
totalndof(::Missing) = missing


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
@assert unshaped(x, shape) == data
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



const _InvValueShape = Base.Fix2{typeof(unshaped),<:AbstractValueShape}

@inline function Base.Broadcast.broadcasted(inv_vs::_InvValueShape, xs)
    Base.Broadcast.broadcasted(unshaped, xs, Ref(inv_vs.x))
end


InverseFunctions.inverse(vs::AbstractValueShape) = Base.Fix2(unshaped, vs)
InverseFunctions.inverse(inv_vs::_InvValueShape) = inv_vs.x

function ChangesOfVariables.with_logabsdet_jacobian(vs::AbstractValueShape, flat_x)
    x = vs(flat_x)
    x, zero(float(eltype(flat_x)))
end

function ChangesOfVariables.with_logabsdet_jacobian(inv_vs::_InvValueShape, x)
    flat_x = inv_vs(x)
    flat_x, zero(float(eltype(flat_x)))
end

const _BroadcastValueShape = Base.Fix1{typeof(broadcast),<:AbstractValueShape}
const _BroadcastInvValueShape = Base.Fix1{typeof(broadcast),<:_InvValueShape}
const _BroadcastUnshaped = Base.Fix1{typeof(broadcast),typeof(unshaped)}

function ChangesOfVariables.with_logabsdet_jacobian(bc_vs::_BroadcastValueShape, ao_flat_x)
    ao_x = bc_vs(ao_flat_x)
    ao_x, zero(float(realnumtype(typeof(ao_flat_x))))
end

function ChangesOfVariables.with_logabsdet_jacobian(bc_inv_vs::Union{_BroadcastInvValueShape,_BroadcastUnshaped}, ao_x)
    ao_flat_x = bc_inv_vs(ao_x)
    ao_flat_x, zero(float(realnumtype(typeof(ao_flat_x))))
end

const _VSTrafo = Union{
    AbstractValueShape, _InvValueShape, typeof(unshaped),
    _BroadcastValueShape, _BroadcastInvValueShape, _BroadcastUnshaped
}

Base.:(∘)(::typeof(identity), f::_VSTrafo) = f
Base.:(∘)(f::_VSTrafo, ::typeof(identity)) = f


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
@assert x1 isa NamedTuple

shape2 = ArrayShape{Real}(3)
x2 = shape2(data)
@assert x2 isa AbstractArray{Int,1}
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
    _apply_accessor_to_data(ValueAccessor(shape, 0), data)
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


# Specialize (::AbstractValueShape).(::AbstractVector{<:AbstractVector{<:Real}}):
Base.Broadcast.broadcasted(vs::AbstractValueShape, A::AbstractArray{<:AbstractVector{<:Real},N}) where N =
    broadcast(view, A, Ref(ValueAccessor(vs, 0)))

# Specialize unshaped for real vectors (semantically vectors of scalar-shaped values)
function Base.Broadcast.broadcasted(::typeof(unshaped), x::AbstractVector{<:Real})
    nestedview(reshape(view(x, :), 1, length(eachindex(x))))
end

function Base.Broadcast.broadcasted(::typeof(unshaped), x::AbstractVector{<:Real}, vsref::Ref{<:AbstractValueShape})
    elshape(x) <= vsref[] || throw(ArgumentError("Shape of value not compatible with given shape"))
    Base.Broadcast.broadcasted(unshaped, x)
end


# Specialize unshaped for real vectors that are array slices:
const _MatrixSliceFirstDim{T} = SubArray{T,1,<:AbstractArray{T,2},<:Tuple{Int,AbstractArray{Int}}}
function Base.Broadcast.broadcasted(::typeof(unshaped), x::_MatrixSliceFirstDim{<:Real})
    nestedview(view(parent(x), x.indices[1]:x.indices[1], x.indices[2]))
end

function Base.Broadcast.broadcasted(::typeof(unshaped), x::_MatrixSliceFirstDim{<:Real}, vsref::Ref{<:AbstractValueShape})
    elshape(x) <= vsref[] || throw(ArgumentError("Shape of value not compatible with given shape"))
    Base.Broadcast.broadcasted(unshaped, x)
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

gradient_shape(vs::AbstractValueShape) = replace_const_shapes(nonstrict_const_zero_shape, vs)
export gradient_shape


"""
    variance_shape(variate_shape::AbstractValueShape)

Return the value shape of the variance of a distribution whose variates have
the value shape `variate_shape`.
"""
function variance_shape end

variance_shape(vs::AbstractValueShape) = replace_const_shapes(const_zero_shape, vs)
export variance_shape
