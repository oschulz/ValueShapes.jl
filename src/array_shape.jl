# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    ArrayShape{T,N} <: AbstractValueShape

Describes the shape of `N`-dimensional arrays of type `T` and a given size.

Constructor:

    ArrayShape{T}(dims::NTuple{N,Integer}) where {T,N}
    ArrayShape{T}(dims::Integer...) where {T}

e.g.

    shape = ArrayShape{Real}(2, 3)

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


@inline _valshapeoftype(T::Type{<:AbstractArray}) = throw(ArgumentError("Type $T does not have a fixed shape"))


@inline default_unshaped_eltype(shape::ArrayShape{T}) where {T} =
    default_unshaped_eltype(_valshapeoftype(T))

@inline shaped_type(shape::ArrayShape{T,N}, ::Type{U}) where {T,N,U<:Real} =
    Array{shaped_type(_valshapeoftype(T),U),N}


@inline function valshape(x::AbstractArray{T}) where T
    _valshapeoftype(T) # ensure T has a fixed shape
    ArrayShape{T}(size(x))
end

# Possible extension: valshape(x::AbstractArrayOfSimilarArrays{...})


totalndof(shape::ArrayShape{T}) where{T} =
    prod(size(shape)) * totalndof(_valshapeoftype(T))


(shape::ArrayShape{T,N})(::UndefInitializer) where {T,N} = Array{default_datatype(T),N}(undef, size(shape)...)


@static if VERSION < v"1.3"
    # Workaround for Julia issue #14919
    @inline (shape::ArrayShape)(data::AbstractVector{<:Real}) =
        _apply_shape_to_data(shape, data)
end



const ArrayAccessor{T,N} = ValueAccessor{ArrayShape{T,N}} where {T,N}

const RealScalarOrVectorAccessor = ValueAccessor{<:Union{ScalarShape{<:Real},ArrayShape{<:Real,1}}}


Base.@propagate_inbounds vs_getindex(data::AbstractVector{<:Real}, va::ArrayAccessor{<:Real}) = copy(view(data, va))

@static if VERSION < v"1.4"
    # To avoid ambiguity with Julia v1.0 (and v1.1 to v1.3?)
    Base.@propagate_inbounds vs_getindex(data::AbstractVector{<:Real}, va::ArrayAccessor{<:Real,1}) = copy(view(data, va))
end

Base.@propagate_inbounds function vs_getindex(
    data::AbstractArray{<:Real,N},
    idxs::Vararg{RealScalarOrVectorAccessor,N}
) where N
    idxs_mapped = map(view_idxs, axes(data), idxs)
    getindex(data, idxs_mapped...)
end


Base.@propagate_inbounds vs_unsafe_view(data::AbstractVector{<:Real}, va::ArrayAccessor{<:Real,1}) where {T} =
    Base.unsafe_view(data, view_idxs(axes(data, 1), va))

Base.@propagate_inbounds vs_unsafe_view(data::AbstractVector{<:Real}, va::ArrayAccessor{<:Real,N}) where {T,N} =
    reshape(Base.unsafe_view(data, view_idxs(axes(data, 1), va)), size(va.shape)...)

Base.@propagate_inbounds function vs_unsafe_view(
    data::AbstractArray{<:Real,N},
    idxs::Vararg{RealScalarOrVectorAccessor,N}
) where N
    idxs_mapped = map(view_idxs, axes(data), idxs)
    Base.view(data, idxs_mapped...)
end



Base.@propagate_inbounds vs_setindex!(data::AbstractVector{<:Real}, v, va::ArrayAccessor{<:Real}) where {T} =
    setindex!(data, v, view_idxs(axes(data, 1), va))

@static if VERSION < v"1.4"
    # To avoid ambiguity with Julia v1.0 (and v1.1 to v1.3?)
    Base.@propagate_inbounds vs_setindex!(data::AbstractVector{<:Real}, v, va::ArrayAccessor{<:Real,1}) where {T} =
        setindex!(data, v, view_idxs(axes(data, 1), va))
end

Base.@propagate_inbounds function vs_setindex!(
    data::AbstractArray{<:Real,N},
    v,
    idxs::Vararg{RealScalarOrVectorAccessor,N}
) where N
    idxs_mapped = map(view_idxs, axes(data), idxs)
    setindex!(data, v, idxs_mapped...)
end


Base.@propagate_inbounds function _bcasted_view(data::AbstractVectorOfSimilarVectors{<:Real}, va::ArrayAccessor{T,1}) where {T,N}
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    fpview = view(flat_data, idxs, :)
    VectorOfSimilarVectors(fpview)
end

Base.@propagate_inbounds function _bcasted_view(data::AbstractVectorOfSimilarVectors{<:Real}, va::ArrayAccessor{T,N}) where {T,N}
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    fpview = view(flat_data, idxs, :)
    VectorOfSimilarArrays(reshape(fpview, size(va.shape)..., :))
end

Base.copy(instance::VSBroadcasted2{1,typeof(getindex),AbstractVectorOfSimilarVectors{<:Real},Ref{<:ArrayAccessor}}) =
    copy(_bcasted_view(instance.args[1], instance.args[2][]))

Base.copy(instance::VSBroadcasted2{1,typeof(view),AbstractVectorOfSimilarVectors{<:Real},Ref{<:ArrayAccessor}}) =
    _bcasted_view(instance.args[1], instance.args[2][])


function _bcasted_view_unchanged(data::AbstractArray{<:AbstractVector{T}}, shape::ArrayShape{U,1}) where {T<:Real,U>:T}
    _checkcompat_inner(shape, data)
    data
end

# Specialize (::ArrayShape{T,1}).(::AbstractArray{<:AbstractVector{<:Real}}):
Base.copy(instance::VSBroadcasted1{N,ArrayShape{T,1},AbstractArray{<:AbstractVector{<:Real},N}}) where {T,N} =
    _bcasted_view_unchanged(instance.args[1], instance.f)


@inline _bcasted_unshaped(A::AbstractArrayOfSimilarArrays{<:Real,1}) = A

# Specialize unshaped.(::AbstractArray{<:AbstractVector{<:Real}}):
Base.copy(instance::VSBroadcasted1{N,typeof(unshaped),AbstractArray{<:AbstractVector{<:Real},N}}) where N =
    _bcasted_unshaped(instance.args[1])


# TODO: Add support for StaticArray.

# Possible extension: variable/flexible array shapes?
