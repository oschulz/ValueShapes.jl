# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    AbstractScalarShape{T} <: AbstractValueShape
"""
abstract type AbstractScalarShape{T} <: AbstractValueShape end

export AbstractScalarShape


@inline Base.eltype(shape::AbstractScalarShape{T}) where {T} = T

@inline Base.size(::AbstractScalarShape) = ()
@inline Base.length(::AbstractScalarShape) = 1



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

(shape::ScalarShape{<:Number})(::UndefInitializer) = zero(nonabstract_eltype(shape))


@static if VERSION < v"1.3"
    # Workaround for Julia issue #14919
    @inline (shape::ScalarShape)(data::AbstractVector{<:Real}) =
        _apply_shape_to_data(shape, data)
end



const ScalarAccessor{T} = ValueAccessor{ScalarShape{T}} where {T}


@inline view_idxs(idxs::AbstractUnitRange{<:Integer}, va::ScalarAccessor{<:Real}) = first(idxs) + va.offset

# ToDo: view_idxs for scalars with dof greater than 1 (complex, etc.)


Base.@propagate_inbounds Base.getindex(data::AbstractVector{<:Real}, va::ScalarAccessor) =
    data[view_idxs(axes(data, 1), va)]

Base.@propagate_inbounds Base.view(data::AbstractVector{<:Real}, va::ScalarAccessor) =
    view(data, view_idxs(axes(data, 1), va))


Base.@propagate_inbounds function _bcasted_getindex(data::AbstractVectorOfSimilarVectors{<:Real}, va::ScalarAccessor)
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    view(flat_data, idxs, :)
end

Base.copy(instance::VSBroadcasted2{typeof(getindex), AbstractVectorOfSimilarVectors{<:Real},Ref{<:ScalarAccessor}}) =
    _bcasted_getindex(instance.args[1], instance.args[2][])    
