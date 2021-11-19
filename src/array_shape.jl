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


import Base.<=
@inline <=(a::ArrayShape{T,N}, b::ArrayShape{U,N}) where {T,U,N} = T<:U && size(a) == size(b)


@inline _valshapeoftype(T::Type{<:AbstractArray}) = throw(ArgumentError("Type $T does not have a fixed shape"))


@inline default_unshaped_eltype(shape::ArrayShape{T}) where {T} =
    default_unshaped_eltype(_valshapeoftype(T))

@inline shaped_type(shape::ArrayShape{T,N}, ::Type{U}) where {T,N,U<:Real} =
    Array{shaped_type(_valshapeoftype(T),U),N}


@inline function valshape(x::AbstractArray{T}) where T
    _valshapeoftype(T) # ensure T has a fixed shape
    ArrayShape{T}(size(x))
end

@inline function valshape(x::AbstractArray{T,0}) where T
    _valshapeoftype(T) # ensure T has a fixed shape
    ScalarShape{T}()
end

# Possible extension: valshape(x::AbstractArrayOfSimilarArrays{...})


totalndof(shape::ArrayShape{T}) where{T} =
    prod(size(shape)) * totalndof(_valshapeoftype(T))


(shape::ArrayShape{T,N})(::UndefInitializer) where {T,N} = Array{default_datatype(T),N}(undef, size(shape)...)


function _check_unshaped_compat(A::AbstractArray{T,N}, shape::ArrayShape{U,N}) where {T<:Real,U<:Real,N}
    Telem = eltype(A)
    Telem <: U || throw(ArgumentError("Element type $Telem of array not compatible with element type $U of given shape"))
    size(A) == size(shape) || throw(ArgumentError("Size of array differs from size of given shape"))
end

function unshaped(A::AbstractArray{T,1}, shape::ArrayShape{U,1}) where {T<:Real,U<:Real}
    _check_unshaped_compat(A, shape)
    A
end

function unshaped(A::AbstractArray{T,N}, shape::ArrayShape{U,N}) where {T<:Real,U<:Real,N}
    _check_unshaped_compat(A, shape)
    reshape(view(A, ntuple(_ -> :, Val(N))...), prod(size(A)))
end

function unshaped(A::Base.ReshapedArray{T,N,<:AbstractArray{T,1}}, shape::ArrayShape{U,N}) where {T<:Real,U<:Real,N}
    _check_unshaped_compat(A, shape)
    parent(A)
end


replace_const_shapes(f::Function, shape::ArrayShape) = shape


@inline function _apply_shape_to_data(shape::ArrayShape{<:Real,1}, data::AbstractVector{<:Real})
    @boundscheck _checkcompat(shape, data)
    data
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


Base.@propagate_inbounds vs_unsafe_view(data::AbstractVector{<:Real}, va::ArrayAccessor{<:Real,1}) =
    Base.unsafe_view(data, view_idxs(axes(data, 1), va))

Base.@propagate_inbounds vs_unsafe_view(data::AbstractVector{<:Real}, va::ArrayAccessor{<:Real,N}) where {N} =
    reshape(Base.unsafe_view(data, view_idxs(axes(data, 1), va)), size(va.shape)...)

Base.@propagate_inbounds function vs_unsafe_view(
    data::AbstractArray{<:Real,N},
    idxs::Vararg{RealScalarOrVectorAccessor,N}
) where N
    idxs_mapped = map(view_idxs, axes(data), idxs)
    Base.view(data, idxs_mapped...)
end



Base.@propagate_inbounds vs_setindex!(data::AbstractVector{<:Real}, v, va::ArrayAccessor{<:Real}) =
    setindex!(data, v, view_idxs(axes(data, 1), va))

@static if VERSION < v"1.4"
    # To avoid ambiguity with Julia v1.0 (and v1.1 to v1.3?)
    Base.@propagate_inbounds vs_setindex!(data::AbstractVector{<:Real}, v, va::ArrayAccessor{<:Real,1}) =
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


Base.@propagate_inbounds function _bcasted_view(data::AbstractVector{<:AbstractVector{<:Real}}, va::ArrayAccessor)
    _bcasted_view(convert(VectorOfSimilarVectors, data), va)
end

Base.@propagate_inbounds function _bcasted_view(data::AbstractVectorOfSimilarVectors{<:Real}, va::ArrayAccessor{T,1}) where {T}
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

Base.Broadcast.broadcasted(::typeof(getindex), A::AbstractVectorOfSimilarVectors{<:Real}, acc::Ref{<:ArrayAccessor}) =
    copy(_bcasted_view(A, acc[]))

Base.Broadcast.broadcasted(::typeof(view), A::AbstractVectorOfSimilarVectors{<:Real}, acc::Ref{<:ArrayAccessor}) =
    copy(_bcasted_view(A, acc[]))


function _bcasted_view_unchanged(data::AbstractArray{<:AbstractVector{T}}, shape::ArrayShape{U,1}) where {T<:Real,U>:T}
    _checkcompat_inner(shape, data)
    data
end

Base.Broadcast.broadcasted(vs::ArrayShape{T,1}, A::AbstractArray{<:AbstractVector{<:Real},N}) where {T,N} =
    _bcasted_view_unchanged(A, vs)


@inline _bcasted_unshaped(A::AbstractArrayOfSimilarVectors{<:Real}) = A
@inline _bcasted_unshaped(A::AbstractArray{<:AbstractVector{<:Real}}) = convert(AbstractArrayOfSimilarVectors, A)

# Specialize unshaped.(::AbstractArray{<:AbstractVector{<:Real}}):
Base.Broadcast.broadcasted(::typeof(unshaped), A::AbstractArray{<:AbstractVector{<:Real}}) =
    _bcasted_unshaped(A)


# TODO: Add support for StaticArray.

# Possible extension: variable/flexible array shapes?
