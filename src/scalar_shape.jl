# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    AbstractScalarShape{T} <: AbstractValueShape
"""
abstract type AbstractScalarShape{T} <: AbstractValueShape end

export AbstractScalarShape


@inline Base.size(::AbstractScalarShape) = ()
@inline Base.length(::AbstractScalarShape) = 1


@inline default_unshaped_eltype(shape::AbstractScalarShape{T}) where {T<:Real} =
    default_datatype(T)

@inline default_unshaped_eltype(shape::AbstractScalarShape{Complex}) = default_datatype(Real)

@inline default_unshaped_eltype(shape::AbstractScalarShape{<:Complex{T}}) where {T} =
    default_unshaped_eltype(_valshapeoftype(T))


@inline shaped_type(shape::AbstractScalarShape{<:Real}, ::Type{T}) where {T<:Real} = T

@inline shaped_type(shape::AbstractScalarShape{<:Complex}, ::Type{T}) where {T<:Real} = Complex{T}



"""
    ScalarShape{T} <: AbstractScalarShape{T}

An `ScalarShape` describes the shape of scalar values of a given type.

Constructor:

    ScalarShape{T::Type}()

T may be an abstract type of Union, or a specific type, e.g.

    ScalarShape{Real}()
    ScalarShape{Integer}()
    ScalarShape{Float32}()
    ScalarShape{Complex}()

Scalar shapes may have a total number of degrees of freedom
(see [`totalndof`](@ref)) greater than one, e.g. shapes of complex-valued
scalars:

    totalndof(ScalarShape{Real}()) == 1
    totalndof(ScalarShape{Complex}()) == 2

See also the documentation of [`AbstractValueShape`](@ref).
"""
struct ScalarShape{T} <: AbstractScalarShape{T} end

export ScalarShape


import Base.<=
@inline <=(a::ScalarShape{T}, b::ScalarShape{U}) where {T,U} = T<:U


@inline _valshapeoftype(T::Type{<:Number}) = ScalarShape{T}()


@inline totalndof(::ScalarShape{T}) where {T <: Real} = 1

@inline function totalndof(::ScalarShape{T}) where {T <: Any}
    if @generated
        fieldtypes = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))
        field_flatlenghts = sum(U -> totalndof(_valshapeoftype(U)), fieldtypes)
        l = prod(field_flatlenghts)
        quote $l end
    else
        fieldtypes = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))
        field_flatlenghts = sum(U -> totalndof(_valshapeoftype(U)), fieldtypes)
        l = prod(field_flatlenghts)
        l
    end
end

(shape::ScalarShape{T})(::UndefInitializer) where {T<:Number} = zero(default_datatype(T))


function unshaped(x::Union{T,AbstractArray{T,0}}, shape::ScalarShape{U}) where {T<:Real,U<:Real}
    T <: U || throw(ArgumentError("Element type $T of scalar value not compatible with type $U of given scalar shape"))
    unshaped(x)
end


replace_const_shapes(f::Function, shape::ScalarShape) = shape



const ScalarAccessor{T} = ValueAccessor{ScalarShape{T}} where {T}


@inline view_idxs(idxs::AbstractUnitRange{<:Integer}, va::ScalarAccessor{<:Real}) = first(idxs) + va.offset

# ToDo: view_idxs for scalars with dof greater than 1 (complex, etc.)


Base.@propagate_inbounds function _bcasted_view(data::AbstractArrayOfSimilarVectors{<:Real,N}, va::ScalarAccessor) where N
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    colons = map(_ -> :, Base.tail(axes(flat_data)))
    view(flat_data, idxs, colons...)
end

Base.copy(instance::VSBroadcasted2{N,typeof(getindex), AbstractArrayOfSimilarVectors{<:Real,N},Ref{<:ScalarAccessor}}) where N =
    copy(_bcasted_view(instance.args[1], instance.args[2][]))

Base.copy(instance::VSBroadcasted2{N,typeof(view), AbstractArrayOfSimilarVectors{<:Real,N},Ref{<:ScalarAccessor}}) where N =
    _bcasted_view(instance.args[1], instance.args[2][])
