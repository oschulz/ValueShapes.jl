# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesChainRulesCoreExt

using ValueShapes

import ChainRulesCore
using ChainRulesCore: AbstractTangent, Tangent, AbstractZero, NoTangent, ZeroTangent
using ChainRulesCore: AbstractThunk, ProjectTo, unthunk, backing


# utils ======================================================================

_backing(x::Any) = x
_backing(x::Tangent) = backing(x)
_unpack_tangent(x::Any) = _backing(unthunk(x))


const _ZeroLike = Union{AbstractZero,Nothing}
const _az_tangent(x::AbstractZero) = x
const _az_tangent(::Nothing) = ZeroTangent()  # Still necessary? Return NoTangent() instead?


_notangent_to_zerotangent(x::Any) = x
_notangent_to_zerotangent(x::Union{NoTangent,Nothing}) = ZeroTangent()
_notangent_to_zerotangent(x::Union{Tuple,NamedTuple}) = map(_notangent_to_zerotangent, x)


function _tangent_array(A::AbstractArray{T}) where T
    dA = similar(A, float(T))
    fill!(dA, NaN) # For safety
    return dA
end

function _tangent_array(A::ArrayOfSimilarVectors{T}) where T
    dA = similar(A, Vector{float(T)})
    fill!(flatview(dA), NaN) # For safety
    return dA
end


# AbstractValueShape =========================================================


vs_cmp_pullback(ΔΩ) = (NoTangent(), NoTangent(), NoTangent())
ChainRulesCore.rrule(::typeof(Base.:(==)), a::AbstractValueShape, b::AbstractValueShape) = (a == b, vs_cmp_pullback)
ChainRulesCore.rrule(::typeof(Base.:(<=)), a::AbstractValueShape, b::AbstractValueShape) = (a <= b, vs_cmp_pullback)
ChainRulesCore.rrule(::typeof(Base.:(>=)), a::AbstractValueShape, b::AbstractValueShape) = (a >= b, vs_cmp_pullback)


# ShapedAsNT =================================================================


# Required for accumulation during automatic differentiation:
function ChainRulesCore.add!!(A::ShapedAsNT{names}, B::ShapedAsNT{names}) where names
    @argcheck _valshape(A) == _valshape(B)
    ChainRulesCore.add!!(_data(A), _data(B))
    return A
end


function ChainRulesCore.Tangent(x::T, unshaped_dx::AbstractVector{<:Real}) where {T<:ShapedAsNT}
    vs = valshape(x)
    gs = gradient_shape(vs)
    contents = (__internal_data = unshaped_dx, __internal_valshape = gs)
    Tangent{T,typeof(contents)}(contents)
end


struct GradShapedAsNTProjector{VS<:NamedTupleShape} <: Function
    gradshape::VS
end

ChainRulesCore.ProjectTo(x::ShapedAsNT) = GradShapedAsNTProjector(gradient_shape(valshape(x)))


_check_ntgs_tangent_compat(a::NamedTupleShape, ::NoTangent) = nothing
function _check_ntgs_tangent_compat(a::NamedTupleShape{names}, b::NamedTupleShape{names}) where names
    a >= b || error("Incompatible tangent NamedTupleShape")
end

_snt_from_tangent(data::AbstractVector{<:Real}, gs::NamedTupleShape) = ShapedAsNT(data, gs)
_snt_from_tangent(data::_ZeroLike, gs::NamedTupleShape) = _az_tangent(data)

function (project::GradShapedAsNTProjector{<:NamedTupleShape{names}})(data::NamedTuple{(:__internal_data, :__internal_valshape)}) where names
    gs = project.gradshape
    _check_ntgs_tangent_compat(gs, data.__internal_valshape)
    _snt_from_tangent(data.__internal_data, project.gradshape)
end

function (project::GradShapedAsNTProjector{<:NamedTupleShape{names}})(tangent::Tangent{<:ShapedAsNT{names}}) where names
    project(_backing(tangent))
end

function (project::GradShapedAsNTProjector{<:NamedTupleShape{names}})(tangent::ShapedAsNT{names}) where names
    tangent
end

(project::GradShapedAsNTProjector{<:NamedTupleShape})(tangent::_ZeroLike) = _az_tangent(tangent)


_getindex_tangent(x::ShapedAsNT, dy::_ZeroLike) = _az_tangent(dy)

function _getindex_tangent(x::ShapedAsNT, dy::NamedTuple)
    tangent = Tangent(x, _tangent_array(unshaped(x)))
    dx_unshaped, gs = _backing(tangent)
    ShapedAsNT(dx_unshaped, gs)[] = _notangent_to_zerotangent(dy)
    tangent
end

function ChainRulesCore.rrule(::typeof(getindex), x::ShapedAsNT)
    shapedasnt_getindex_pullback(ΔΩ) = (NoTangent(), ProjectTo(x)(_getindex_tangent(x, _unpack_tangent(ΔΩ))))
    return x[], shapedasnt_getindex_pullback
end


_unshaped_tangent(x::ShapedAsNT, dy::AbstractArray{<:Real}) = Tangent(x, dy)
_unshaped_tangent(x::ShapedAsNT, dy::_ZeroLike) = _az_tangent(dy)

function ChainRulesCore.rrule(::typeof(unshaped), x::ShapedAsNT)
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), ProjectTo(x)(_unshaped_tangent(x, _unpack_tangent(ΔΩ))))
    return unshaped(x), unshaped_nt_pullback
end

function ChainRulesCore.rrule(::typeof(unshaped), x::ShapedAsNT, vs::NamedTupleShape)
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), ProjectTo(x)(_unshaped_tangent(x, _unpack_tangent(ΔΩ))), NoTangent())
    return unshaped(x, vs), unshaped_nt_pullback
end

function _unshaped_tangent(x::NamedTuple, vs::NamedTupleShape, dy::AbstractArray{<:Real})
    gs = gradient_shape(vs)
    # gs(dy) can be a NamedTuple or a ShapedAsNT, depending on vs:
    dx = convert(NamedTuple, gs(dy))
    Tangent{typeof(x),typeof(dx)}(dx)
end

_unshaped_tangent(x::NamedTuple, vs::NamedTupleShape, dy::_ZeroLike) = _az_tangent(dy)

function ChainRulesCore.rrule(::typeof(unshaped), x::NamedTuple, vs::NamedTupleShape)
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), _unshaped_tangent(x, vs, _unpack_tangent(ΔΩ)), NoTangent())
    return unshaped(x, vs), unshaped_nt_pullback
end


_shapedasnt_tangent(dy::_ZeroLike, vs::NamedTupleShape{names}) where names = _az_tangent(dy)

_shapedasnt_tangent(dy::ShapedAsNT{names}, vs::NamedTupleShape{names}) where names = unshaped(dy)

function _shapedasnt_tangent(
    dy::Tangent{<:NamedTuple{names},<:NamedTuple{names}},
    vs::NamedTupleShape{names}
) where names
    unshaped(_backing(dy), gradient_shape(vs))
end

function _shapedasnt_tangent(
    dy::Tangent{<:Any,<:NamedTuple{(:__internal_data, :__internal_valshape)}},
    vs::NamedTupleShape{names}
) where names
    _backing(dy).__internal_data
end

function ChainRulesCore.rrule(::Type{ShapedAsNT}, A::AbstractVector{<:Real}, vs::NamedTupleShape{names}) where names
    shapedasnt_pullback(ΔΩ) = (NoTangent(), _shapedasnt_tangent(unthunk(ΔΩ), vs), NoTangent())
    return ShapedAsNT(A, vs), shapedasnt_pullback
end


# ShapedAsNTArray ============================================================


# For accumulation during automatic differentiation:
function ChainRulesCore.add!!(A::_AnySNTArray{names}, B::_AnySNTArray{names}) where names
    @argcheck elshape(A) == elshape(B)
    ChainRulesCore.add!!(_data(A), _data(B))
    return A
end


function ChainRulesCore.Tangent(X::T, unshaped_dX::AbstractArray{<:AbstractVector{<:Real}}) where {T<:ShapedAsNTArray}
    vs = elshape(X)
    gs = gradient_shape(vs)
    contents = (__internal_data = unshaped_dX, __internal_elshape = gs)
    Tangent{T,typeof(contents)}(contents)
end


struct GradShapedAsNTArrayProjector{VS<:NamedTupleShape} <: Function
    gradshape::VS
end

ChainRulesCore.ProjectTo(X::ShapedAsNTArray) = GradShapedAsNTArrayProjector(gradient_shape(elshape(X)))

function (project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(data::NamedTuple{(:__internal_data, :__internal_elshape)}) where names
    _check_ntgs_tangent_compat(project.gradshape, data.__internal_elshape)
    _keep_zerolike(ShapedAsNTArray, data.__internal_data, project.gradshape)
end

@inline _keep_zerolike(::Type{T}, x, xs...) where T = T(x, xs...)
@inline _keep_zerolike(::Type{T}, x::_ZeroLike, xs...) where T = _az_tangent(x)
@inline _keep_zerolike(f::F, x, xs...) where F = f(x, xs...)
@inline _keep_zerolike(f::F, x::_ZeroLike, xs...) where F = _az_tangent(x)


(project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(tangent::Tangent{<:_AnySNTArray}) where names = project(_backing(tangent))
(project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(tangent::_AnySNTArray) where names = tangent
(project::GradShapedAsNTArrayProjector{<:NamedTupleShape})(tangent::_ZeroLike) = _az_tangent(tangent)

function (project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(
    tangent::AbstractArray{<:Union{Tangent{<:Any,<:NamedTuple{names}},ShapedAsNT{names}}}
) where names
    data =_shapedasntarray_tangent(tangent, project.gradshape)
    ShapedAsNTArray(data, project.gradshape)
end


_tablecols_tangent(X::ShapedAsNTArray, dY::_ZeroLike) = _az_tangent(dY)

function _write_snta_col!(data::AbstractArray{<:AbstractVector{<:Real}}, va::ValueAccessor, A::AbstractArray)
    B = view.(data, Ref(va))
    B .= A
end
function _write_snta_col!(data::ArrayOfSimilarVectors{<:Real}, va::ValueAccessor, A::_ZeroLike)
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    fill!(view(flat_data, idxs, :), zero(eltype(flat_data)))
end
_write_snta_col!(data::AbstractArray{<:AbstractVector{<:Real}}, va::ConstAccessor, A) = nothing

function _tablecols_tangent(X::_AnySNTArray, dY::NamedTuple{names}) where names
    tangent = Tangent(X, _tangent_array(_data(X)))
    dx_unshaped, gs = _backing(tangent)
    # ToDo: Re-check safety of this after nested-NT arrays are implemented:
    map((va_i, dY_i) -> _write_snta_col!(dx_unshaped, va_i, dY_i), _accessors(gs), _notangent_to_zerotangent(dY))
    tangent
end

function ChainRulesCore.rrule(::typeof(Tables.columns), X::ShapedAsNTArray)
    tablecols_pullback(ΔΩ) = begin
        (NoTangent(), ProjectTo(X)(_tablecols_tangent(X, _unpack_tangent(ΔΩ))))
    end
    return Tables.columns(X), tablecols_pullback
end


_data_tangent(X::ShapedAsNTArray, dY::AbstractArray{<:AbstractVector{<:Real}}) = Tangent(X, dY)
_data_tangent(X::ShapedAsNTArray, dY::_ZeroLike) = _az_tangent(dY)

function ChainRulesCore.rrule(::typeof(_data), X::ShapedAsNTArray)
    _data_pullback(ΔΩ) = (NoTangent(), ProjectTo(X)(_data_tangent(X, _unpack_tangent(ΔΩ))))
    return _data(X), _data_pullback
end


_shapedasntarray_tangent(dY::_ZeroLike, vs::NamedTupleShape{names}) where names = _az_tangent(dY)

_shapedasntarray_tangent(dY::_AnySNTArray, vs::NamedTupleShape{names}) where names = unshaped.(dY)

function _shapedasntarray_tangent(
    dY::AbstractArray{<:Tangent{<:Any,<:NamedTuple{names}}},
    vs::NamedTupleShape{names}
) where names
    ArrayOfSimilarArrays(unshaped.(ValueShapes._backing.(dY), Ref(gradient_shape(vs))))
end

function _shapedasntarray_tangent(
    dY::AbstractArray{<:ShapedAsNT{names}},
    vs::NamedTupleShape{names}
) where names
    ArrayOfSimilarArrays(unshaped.(dY))
end

function _shapedasntarray_tangent(
    dY::Tangent{<:Any,<:NamedTuple{(:__internal_data, :__internal_elshape)}},
    vs::NamedTupleShape{names}
) where names
    _backing(dY).__internal_data
end

function ChainRulesCore.rrule(::Type{ShapedAsNTArray}, A::AbstractArray{<:AbstractVector{<:Real}}, vs::NamedTupleShape{names}) where names
    shapedasntarray_pullback(ΔΩ) = begin
        (NoTangent(), _shapedasntarray_tangent(unthunk(ΔΩ), vs), NoTangent())
    end
    return ShapedAsNTArray(A, vs), shapedasntarray_pullback
end


end # module ValueShapesChainRulesCoreExt
