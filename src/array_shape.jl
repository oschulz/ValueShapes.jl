# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    ArrayShape{T,N} <: AbstractValueShape

Describes the shape of `N`-dimensional arrays of type `T` and a given size.

Constructor:

    ArrayShape{T}(dims::NTuple{N,Integer}) where {T,N}
    ArrayShape{T}(dims::Integer...) where {T}

e.g.

    shape = ArrayShape{Real}(2, 3)

In addition to using the shape as a value constructor

    size(shape(undef)) == (2, 3)
    eltype(shape(undef)) == Float64

(see [`AbstractValueShape`](@ref)), a shape can also be used as an argument of
certains array type constructors to explicitly construct standard `Array`s
or `ElasticArray`s:

    using ElasticArrays

    size(Array(undef, shape)) == (2, 3)
    eltype(Array(undef, shape)) == Float64

    size(ElasticArray(undef, shape)) == (2, 3)

See also the documentation of [`AbstractValueShape`](@ref).
"""
struct ArrayShape{T,N} <: AbstractValueShape
    dims::NTuple{N,Int}
end

export ArrayShape


ArrayShape{T}(dims::NTuple{N,Integer}) where {T,N} = ArrayShape{T,N}(map(Int, dims))
ArrayShape{T}(dims::Integer...) where {T} = ArrayShape{T}(dims)


@inline Base.size(shape::ArrayShape) = shape.dims
Base.length(shape::ArrayShape) = prod(size(shape))

@inline Base.eltype(::ArrayShape{T}) where {T} = T


@inline _valshapeoftype(T::Type{<:AbstractArray}) = throw(ArgumentError("Type $T does not have a fixed shape"))

@inline function valshape(x::AbstractArray{T}) where T
    _valshapeoftype(T) # ensure T has a fixed shape
    ArrayShape{T}(size(x))
end

# Possible extension: valshape(x::AbstractArrayOfSimilarArrays{...})


totalndof(shape::ArrayShape{T}) where{T} =
    prod(size(shape)) * totalndof(_valshapeoftype(T))


(shape::ArrayShape{T,N})(::UndefInitializer) where {T,N} = Array{nonabstract_eltype(shape),N}(undef, size(shape)...)


@static if VERSION < v"1.3"
    # Workaround for Julia issue #14919
    @inline (shape::ArrayShape)(data::AbstractVector{<:Real}) =
        _apply_shape_to_data(shape, data)
end


@inline Array{U}(::UndefInitializer, shape::ArrayShape{T}) where {T,U<:T} =
    Array{U}(undef, size(shape)...)

@inline Array(::UndefInitializer, shape::ArrayShape) =
    Array{nonabstract_eltype(shape)}(undef, shape)


@inline ElasticArray{U}(::UndefInitializer, shape::ArrayShape{T}) where {T,U<:T} =
    ElasticArray{U}(undef, size(shape)...)

@inline ElasticArray(::UndefInitializer, shape::ArrayShape) =
    ElasticArray{nonabstract_eltype(shape)}(undef, shape)



const ArrayAccessor{T,N} = ValueAccessor{ArrayShape{T,N}} where {T,N}


Base.@propagate_inbounds vs_getindex(data::AbstractVector{<:Real}, va::ArrayAccessor) = view(data, va)

Base.@propagate_inbounds vs_unsafe_view(data::AbstractVector{<:Real}, va::ArrayAccessor{T,1}) where {T} =
    Base.unsafe_view(data, view_idxs(axes(data, 1), va))

Base.@propagate_inbounds vs_unsafe_view(data::AbstractVector{<:Real}, va::ArrayAccessor{T,N}) where {T,N} =
    reshape(Base.unsafe_view(data, view_idxs(axes(data, 1), va)), size(va.shape)...)


Base.@propagate_inbounds vs_setindex!(data::AbstractVector{<:Real}, v, va::ArrayAccessor) where {T} =
    setindex!(data, v, view_idxs(axes(data, 1), va))


Base.@propagate_inbounds function _bcasted_getindex(data::AbstractVectorOfSimilarVectors{<:Real}, va::ArrayAccessor{T,1}) where {T,N}
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    fpview = view(flat_data, idxs, :)
    VectorOfSimilarVectors(fpview)
end

Base.@propagate_inbounds function _bcasted_getindex(data::AbstractVectorOfSimilarVectors{<:Real}, va::ArrayAccessor{T,N}) where {T,N}
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    fpview = view(flat_data, idxs, :)
    VectorOfSimilarArrays(reshape(fpview, size(va.shape)..., :))
end

Base.copy(instance::VSBroadcasted2{typeof(getindex),AbstractVectorOfSimilarVectors{<:Real},Ref{<:ArrayAccessor}}) =
    _bcasted_getindex(instance.args[1], instance.args[2][])    


# TODO: Add support for StaticArray.

# Possible extension: variable/flexible array shapes?
