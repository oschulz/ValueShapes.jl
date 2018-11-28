# This file is a part of ShapesOfVariables.jl, licensed under the MIT License (MIT).


abstract type AbstractValueShape end

export AbstractValueShape


function shapeof end
export shapeof

function flatdof end
export flatdof


Base.@pure shapeof(T::Type{<:Any}) = ScalarShape{T}()

Base.@pure shapeof(T::Type{<:AbstractArray}) = throw(ArgumentError("Type $T does not have a fixed shape"))

Base.@pure shapeof(::T) where {T<:Number} = shapeof(T)

@inline shapeof(x::AbstractArray{<:T}) where T = ArrayShape{T}(size(x))


Base.@pure Base.convert(::Type{AbstractValueShape}, T::Type{<:Any}) = shapeof(T)



struct ScalarShape{T} <: AbstractValueShape end

export ScalarShape


Base.@pure Base.eltype(shape::ScalarShape{T}) where {T} = T

Base.@pure Base.size(::ScalarShape) = 1



Base.@pure flatdof(::ScalarShape{T}) where {T <: Real} = 1

Base.@pure @generated function flatdof(::ScalarShape{T}) where {T <: Any}
    fieldtypes = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))
    field_flatlenghts = sum(U -> flatdof(shapeof(U)), fieldtypes)
    l = prod(field_flatlenghts)
    quote $l end
end



struct ArrayShape{T,N} <: AbstractValueShape
    dims::NTuple{N,Int}
end

export ArrayShape


ArrayShape{T}(dims::NTuple{N,Integer}) where {T,N} = ArrayShape{T,N}(map(Int, dims))
ArrayShape{T}(dims::Integer...) where {T} = ArrayShape{T}(dims)


@inline Base.size(shape::ArrayShape) = shape.dims

Base.@pure Base.eltype(::ArrayShape{T}) where {T} = T

flatdof(shape::ArrayShape{T}) where{T} =
    prod(size(shape)) * flatdof(convert(AbstractValueShape, T))


@inline Base.convert(::Type{AbstractValueShape}, tuple::Tuple) = _tupleentries_to_valshape(tuple...)

@inline _tupleentries_to_valshape() where {T} = ScalarShape{Real}()

@inline _tupleentries_to_valshape(dim1::Integer, dims::Integer...) = ArrayShape{Real}(dim1, dims...)


@inline Array{U}(::UndefInitializer, shape::ArrayShape{T}) where {T,U<:T} =
    Array{U}(undef, size(shape)...)

@inline Array(::UndefInitializer, shape::ArrayShape) =
    Array{nonabstract_eltype(shape)}(undef, shape)


@inline ElasticArray{U}(::UndefInitializer, shape::ArrayShape{T}) where {T,U<:T} =
    ElasticArray{U}(undef, size(shape)...)

@inline ElasticArray(::UndefInitializer, shape::ArrayShape) =
    ElasticArray{nonabstract_eltype(shape)}(undef, shape)


function nonabstract_eltype end
export nonabstract_eltype

@inline function nonabstract_eltype(shape::AbstractValueShape)
    T = eltype(shape)
    if (isabstracttype(T))
        auto_nonabstract(T)
    else
        T
    end
end

Base.@pure auto_nonabstract(::Type{>:Int}) = Int
Base.@pure auto_nonabstract(::Type{>:Float64}) = Float64
Base.@pure auto_nonabstract(::Type{>:Real}) = Float64


Base.@pure _multi_promote_type() = Nothing
Base.@pure _multi_promote_type(T::Type) = T
Base.@pure _multi_promote_type(T::Type, U::Type, rest::Type...) = promote_type(T, _multi_promote_type(U, rest...))



const ScalarValueInstance = Union{Number,String,Symbol}


struct ConstValueShape{T} <: AbstractValueShape
    value::T
end

@inline Base.size(shape::ConstValueShape) = size(shape.value)

@inline Base.eltype(shape::ConstValueShape) = eltype(shape.value)

Base.@pure flatdof(::ConstValueShape) = 0

@inline Base.convert(::Type{AbstractValueShape}, x::ScalarValueInstance) = ConstValueShape(x)
@inline Base.convert(::Type{AbstractValueShape}, x::AbstractArray{<:ScalarValueInstance}) = ConstValueShape(x)
