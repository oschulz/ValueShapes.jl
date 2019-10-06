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

An `AbstractValueShape` combines type and size information, the combination of
which is termed shape, here. Subtypes are defined for shapes of scalars
(see [`ScalarShape`](@ref)), arrays (see [`ArrayShape`](@ref)) and constant
values (see [`ConstValueShape`](@ref)).

Subtype of `AbstractValueShape` must support `eltype`, `size` and
[`totalndof`](@ref).
"""
abstract type AbstractValueShape end

export AbstractValueShape


Base.length(shape::AbstractValueShape) = prod(size(shape))


function shapeoftype end
export shapeoftype

@inline shapeoftype(T::Type{<:Any}) = ScalarShape{T}()

@inline shapeoftype(T::Type{<:AbstractArray}) = throw(ArgumentError("Type $T does not have a fixed shape"))


function shapeof end
export shapeof

@inline shapeof(x::T) where T = shapeoftype(T)

@inline function shapeof(x::AbstractArray{T}) where T
    shapeoftype(T) # ensure T has a fixed shape
    ArrayShape{T}(size(x))
end

# Possible extension: shapeof(x::AbstractArrayOfSimilarArrays{...})


"""
    totalndof(shape::AbstractValueShape)

Get the total number of degrees of freedom of values having the given shape.

Equivalent to the size of the array required when flattening values of this
shape into an array of real numbers, without including any constant values.
"""
function totalndof end
export totalndof


@inline nonabstract_eltype(shape::AbstractValueShape) = default_datatype(eltype(shape))



"""
    ScalarShape{T} <: AbstractValueShape

An `ScalarShape` describes the shape of scalar values of a given type.

Constructor:

    ScalarShape{T::Type}()

T may be an abstract type of Union, or a specific type, e.g.

    ScalarShape{Integer}()
    ScalarShape{Real}()
    ScalarShape{Complex}()
    ScalarShape{Float32}()

Scalar shapes may have a total number of degrees of freedom
(see [`totalndof`](@ref)) greater than one, e.g. shapes of complex-valued
scalars:

    totalndof(ScalarShape{Real}()) == 1
    totalndof(ScalarShape{Complex}()) == 2
"""
struct ScalarShape{T} <: AbstractValueShape end

export ScalarShape


Base.@pure Base.eltype(shape::ScalarShape{T}) where {T} = T

Base.@pure Base.size(::ScalarShape) = 1


Base.@pure totalndof(::ScalarShape{T}) where {T <: Real} = 1

@inline function totalndof(::ScalarShape{T}) where {T <: Any}
    if @generated
        fieldtypes = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))
        field_flatlenghts = sum(U -> totalndof(shapeoftype(U)), fieldtypes)
        l = prod(field_flatlenghts)
        quote $l end
    else
        fieldtypes = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))
        field_flatlenghts = sum(U -> totalndof(shapeoftype(U)), fieldtypes)
        l = prod(field_flatlenghts)
        l
    end
end



"""
    ArrayShape{T,N} <: AbstractValueShape

Describes the shape of `N`-dimensional arrays of type `T` and a given size.

Constructor:

    ArrayShape{T}(dims::NTuple{N,Integer}) where {T,N}
    ArrayShape{T}(dims::Integer...) where {T}

e.g.

    shape = ArrayShape{Real}(2, 3)

Array shapes can be used to instantiate array of the given shape, e.g.

    size(Array(undef, shape)) == (2, 3)
    size(ElasticArrays.ElasticArray(undef, shape)) == (2, 3)

If the element type of the shape of an abstract type of union,
[`ValueShapes.default_datatype`](@ref) will be used to determine a
suitable more specific type, if possible:

    eltype(Array(undef, shape)) == Float64
"""
struct ArrayShape{T,N} <: AbstractValueShape
    dims::NTuple{N,Int}
end

export ArrayShape


ArrayShape{T}(dims::NTuple{N,Integer}) where {T,N} = ArrayShape{T,N}(map(Int, dims))
ArrayShape{T}(dims::Integer...) where {T} = ArrayShape{T}(dims)


@inline Base.size(shape::ArrayShape) = shape.dims

Base.@pure Base.eltype(::ArrayShape{T}) where {T} = T

totalndof(shape::ArrayShape{T}) where{T} =
    prod(size(shape)) * totalndof(shapeoftype(T))


@inline Array{U}(::UndefInitializer, shape::ArrayShape{T}) where {T,U<:T} =
    Array{U}(undef, size(shape)...)

@inline Array(::UndefInitializer, shape::ArrayShape) =
    Array{nonabstract_eltype(shape)}(undef, shape)


@inline ElasticArray{U}(::UndefInitializer, shape::ArrayShape{T}) where {T,U<:T} =
    ElasticArray{U}(undef, size(shape)...)

@inline ElasticArray(::UndefInitializer, shape::ArrayShape) =
    ElasticArray{nonabstract_eltype(shape)}(undef, shape)


# TODO: Add support for StaticArray.



"""
    ConstValueShape{T} <: AbstractValueShape

A `ConstValueShape` describes the shape of constant values of type `T`.

Constructor:

    ConstValueShape(value)

`value` may be of arbitrary type, e.g. a constant scalar value or array:

    ConstValueShape(4.2),
    ConstValueShape([11 21; 12 22]),

Shapes of constant values have zero degrees of freedom
((see [`totalndof`](@ref)).
"""
struct ConstValueShape{T} <: AbstractValueShape
    value::T
end

export ConstValueShape


@inline Base.size(shape::ConstValueShape) = size(shape.value)

@inline Base.eltype(shape::ConstValueShape) = eltype(shape.value)

Base.@pure totalndof(::ConstValueShape) = 0



# Possible extension: variable/flexible array shapes?
