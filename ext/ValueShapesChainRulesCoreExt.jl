# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesChainRulesCoreExt

using ValueShapes
using ValueShapes: _adignore_call, _data, _valshape, _accessors, _AnySNTArray,
    NamedTupleAccessor, _flat_view

using ArgCheck
using ArraysOfArrays
using Distributions: UnivariateDistribution

import ChainRulesCore
import Tables

using ChainRulesCore: AbstractTangent, Tangent, AbstractZero, NoTangent, ZeroTangent
using ChainRulesCore: AbstractThunk, ProjectTo, unthunk, backing, canonicalize


# tangent utils ==============================================================


@inline _adignore_call_pullback(@nospecialize ΔΩ) = (NoTangent(), NoTangent())
ChainRulesCore.rrule(::typeof(_adignore_call), f) = _adignore_call(f), _adignore_call_pullback


_backing(x::Any) = x
_backing(x::Tangent) = backing(canonicalize(x))
_unpack_tangent(x::Any) = _backing(unthunk(x))


const _ZeroLike = Union{AbstractZero,Nothing}
const _az_tangent(x::AbstractZero) = x
const _az_tangent(::Nothing) = ZeroTangent()  # Still necessary? Return NoTangent() instead?


# Resolves thunks and structural tangents recursively and turns `nothing`
# into `ZeroTangent`. Arrays of real numbers and shaped values are left as
# they are:
_normalize_tangent(x::Any) = x
_normalize_tangent(x::AbstractThunk) = _normalize_tangent(unthunk(x))
_normalize_tangent(::Nothing) = ZeroTangent()
_normalize_tangent(x::Tangent) = _normalize_tangent(_backing(x))
_normalize_tangent(x::Union{Tuple,NamedTuple}) = map(_normalize_tangent, x)
_normalize_tangent(x::AbstractArray) = map(_normalize_tangent, x)
_normalize_tangent(x::AbstractArray{<:Real}) = x
_normalize_tangent(x::AbstractArray{<:AbstractArray{<:Real}}) = x
_normalize_tangent(x::_AnySNTArray) = x


# Real type of a normalized tangent, for allocating flat tangent arrays:
_tangent_realtype(::Any) = Bool
_tangent_realtype(x::Real) = typeof(x)
_tangent_realtype(x::AbstractArray{<:Real}) = eltype(x)
_tangent_realtype(x::AbstractArray{<:AbstractArray{<:Real}}) = eltype(eltype(x))
_tangent_realtype(x::AbstractArray) = mapreduce(_tangent_realtype, promote_type, x, init = Bool)
_tangent_realtype(x::ShapedAsNT) = eltype(unshaped(x))
_tangent_realtype(x::_AnySNTArray) = eltype(eltype(_data(x)))
_tangent_realtype(x::Union{Tuple,NamedTuple}) = promote_type(Bool, map(_tangent_realtype, values(x))...)

_tangent_real(x::Real) = x
_tangent_real(::AbstractZero) = false


_has_internal_data(::Type{<:NamedTuple{names}}) where names = Base.sym_in(:__internal_data, names)

# A NamedTuple tangent of a nested value is structural (carries the flat data
# of a ShapedAsNT) if the value is a ShapedAsNT by the value type of its shape
# and the tangent has the internal data field. NamedTuple values get logical
# tangents, a structural one is only recognized if the shape has no field of
# that name:
_structural_tangent(::Type{DY}, ::Type{<:NamedTupleShape{names,AT,<:ShapedAsNT}}) where {DY<:NamedTuple,names,AT} =
    _has_internal_data(DY)
_structural_tangent(::Type{DY}, ::Type{<:NamedTupleShape{names,AT,<:NamedTuple}}) where {DY<:NamedTuple,names,AT} =
    _has_internal_data(DY) && !Base.sym_in(:__internal_data, names)


# Writes a normalized tangent dy of a value of shape vs into the flat tangent
# vector dx. Omitted fields of structural tangents are zero:
_write_tangent!(dx::AbstractVector{<:Real}, dy::AbstractZero, vs::AbstractValueShape) = fill!(dx, zero(eltype(dx)))
_write_tangent!(dx::AbstractVector{<:Real}, dy, vs::ConstValueShape) = dx
_write_tangent!(dx::AbstractVector{<:Real}, dy::AbstractZero, vs::ConstValueShape) = dx
_write_tangent!(dx::AbstractVector{<:Real}, dy::Real, vs::ScalarShape) = fill!(dx, dy)
_write_tangent!(dx::AbstractVector{<:Real}, dy::AbstractArray{<:Any,0}, vs::ScalarShape) = fill!(dx, _tangent_real(dy[]))
_write_tangent!(dx::AbstractVector{<:Real}, dy::AbstractArray, vs::ArrayShape) = (dx .= _tangent_real.(vec(dy)); dx)
_write_tangent!(dx::AbstractVector{<:Real}, dy::AbstractVector{<:Real}, vs::NamedTupleShape) = copyto!(dx, dy)
_write_tangent!(dx::AbstractVector{<:Real}, dy::ShapedAsNT, vs::NamedTupleShape) = copyto!(dx, unshaped(dy))

# Logical tangent of a NamedTuple value, omitted fields are zero:
function _write_fields_tangent!(dx::AbstractVector{<:Real}, dy::NamedTuple, vs::NamedTupleShape{names}) where names
    zero_fields = NamedTuple{names}(ntuple(_ -> ZeroTangent(), Val(length(names))))
    dy_complete = NamedTuple{names}(merge(zero_fields, dy))
    map(values(_accessors(vs)), values(dy_complete)) do acc, dy_acc
        _write_tangent!(_flat_view(dx, acc), dy_acc, acc.shape)
    end
    dx
end

# Structural tangent of a ShapedAsNT value, carries the flat data:
_write_structural_tangent!(dx::AbstractVector{<:Real}, dy::NamedTuple, vs::NamedTupleShape) =
    _write_tangent!(dx, dy.__internal_data, vs)

# Nested values, with the representation given by the value type of their shape:
_write_tangent!(dx::AbstractVector{<:Real}, dy::NamedTuple, vs::NamedTupleShape) =
    _structural_tangent(typeof(dy), typeof(vs)) ? _write_structural_tangent!(dx, dy, vs) : _write_fields_tangent!(dx, dy, vs)

# Tangents of a value of known representation, a ShapedAsNT or its NamedTuple value:
_write_snt_tangent!(dx::AbstractVector{<:Real}, dy, vs::NamedTupleShape) = _write_tangent!(dx, dy, vs)
_write_snt_tangent!(dx::AbstractVector{<:Real}, dy::NamedTuple, vs::NamedTupleShape) =
    _has_internal_data(typeof(dy)) ? _write_structural_tangent!(dx, dy, vs) : _write_fields_tangent!(dx, dy, vs)

_write_nt_tangent!(dx::AbstractVector{<:Real}, dy, vs::NamedTupleShape) = _write_tangent!(dx, dy, vs)
_write_nt_tangent!(dx::AbstractVector{<:Real}, dy::NamedTuple, vs::NamedTupleShape) = _write_fields_tangent!(dx, dy, vs)


# Flat tangent for primal data A from a normalized tangent dy of the value
# ShapedAsNT(A, vs) (write_nt! = _write_snt_tangent!) or of its NamedTuple
# value (write_nt! = _write_nt_tangent!). Natural tangents and structural flat
# data pass through without copying:
_flat_tangent(write_nt!, dy::AbstractZero, vs::NamedTupleShape, A::AbstractVector{<:Real}) = dy
_flat_tangent(write_nt!, dy::ShapedAsNT, vs::NamedTupleShape, A::AbstractVector{<:Real}) = unshaped(dy)
_flat_tangent(write_nt!, dy::AbstractVector{<:Real}, vs::NamedTupleShape, A::AbstractVector{<:Real}) = dy

function _flat_tangent(write_nt!, dy::NamedTuple, vs::NamedTupleShape, A::AbstractVector{<:Real})
    if write_nt! === _write_snt_tangent! && _has_internal_data(typeof(dy))
        _flat_tangent(write_nt!, dy.__internal_data, vs, A)
    else
        dx = similar(A, promote_type(float(eltype(A)), _tangent_realtype(dy)))
        write_nt!(dx, dy, vs)
    end
end


@inline _keep_zerolike(::Type{T}, x, xs...) where T = T(x, xs...)
@inline _keep_zerolike(::Type{T}, x::_ZeroLike, xs...) where T = _az_tangent(x)
@inline _keep_zerolike(f::F, x, xs...) where F = f(x, xs...)
@inline _keep_zerolike(f::F, x::_ZeroLike, xs...) where F = _az_tangent(x)


# AbstractValueShape =========================================================


vs_cmp_pullback(ΔΩ) = (NoTangent(), NoTangent(), NoTangent())
ChainRulesCore.rrule(::typeof(Base.:(==)), a::AbstractValueShape, b::AbstractValueShape) = (a == b, vs_cmp_pullback)
ChainRulesCore.rrule(::typeof(Base.:(<=)), a::AbstractValueShape, b::AbstractValueShape) = (a <= b, vs_cmp_pullback)
ChainRulesCore.rrule(::typeof(Base.:(>=)), a::AbstractValueShape, b::AbstractValueShape) = (a >= b, vs_cmp_pullback)


# Distributions ==============================================================


_unshaped_uv_pullback(ΔΩ) = NoTangent(), only(unthunk(ΔΩ).v)
ChainRulesCore.rrule(::typeof(unshaped), d::UnivariateDistribution) = unshaped(d), _unshaped_uv_pullback


# ShapedAsNT =================================================================


Base.@propagate_inbounds function Base.setproperty!(A::ShapedAsNT, p::Symbol, x::AbstractZero)
    data = _data(A)
    va = getproperty(_accessors(_valshape(A)), p)
    fill!(_flat_view(data, va), zero(eltype(data)))
    A
end


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

# Tangent of a ShapedAsNT with gradient shape gs, from its flat tangent:
_snt_tangent(dx::AbstractZero, gs::NamedTupleShape) = dx
_snt_tangent(dx::AbstractVector{<:Real}, gs::NamedTupleShape) = ShapedAsNT(dx, gs)


struct GradShapedAsNTProjector{VS<:NamedTupleShape} <: Function
    gradshape::VS
end

ChainRulesCore.ProjectTo(x::ShapedAsNT) = GradShapedAsNTProjector(gradient_shape(valshape(x)))


_check_ntgs_tangent_compat(a::NamedTupleShape, ::_ZeroLike) = nothing
function _check_ntgs_tangent_compat(a::NamedTupleShape{names}, b::NamedTupleShape{names}) where names
    a >= b || error("Incompatible tangent NamedTupleShape")
end

_snt_from_tangent(data::AbstractVector{<:Real}, gs::NamedTupleShape) = ShapedAsNT(data, gs)
_snt_from_tangent(data::_ZeroLike, ::NamedTupleShape) = _az_tangent(data)
_snt_from_tangent(data::AbstractThunk, gs::NamedTupleShape) = _snt_from_tangent(unthunk(data), gs)

function (project::GradShapedAsNTProjector)(data::NamedTuple)
    gs = project.gradshape
    if _has_internal_data(typeof(data))
        _check_ntgs_tangent_compat(gs, get(data, :__internal_valshape, NoTangent()))
        _snt_from_tangent(data.__internal_data, gs)
    else
        dy = _normalize_tangent(data)
        dx = Vector{promote_type(float(ValueShapes.default_unshaped_eltype(gs)), _tangent_realtype(dy))}(undef, totalndof(gs))
        ShapedAsNT(_write_fields_tangent!(dx, dy, gs), gs)
    end
end

(project::GradShapedAsNTProjector)(tangent::Tangent) = project(_backing(tangent))
(project::GradShapedAsNTProjector)(tangent::AbstractThunk) = project(unthunk(tangent))
(project::GradShapedAsNTProjector{<:NamedTupleShape{names}})(tangent::ShapedAsNT{names}) where names = tangent
(project::GradShapedAsNTProjector)(tangent::_ZeroLike) = _az_tangent(tangent)


function ChainRulesCore.rrule(::typeof(getindex), x::ShapedAsNT)
    vs = valshape(x)
    gs = gradient_shape(vs)
    function shapedasnt_getindex_pullback(ΔΩ)
        dx = _flat_tangent(_write_nt_tangent!, _normalize_tangent(ΔΩ), vs, unshaped(x))
        (NoTangent(), _snt_tangent(dx, gs))
    end
    return x[], shapedasnt_getindex_pullback
end


function ChainRulesCore.rrule(::typeof(unshaped), x::ShapedAsNT)
    gs = gradient_shape(valshape(x))
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), _snt_tangent(_normalize_tangent(ΔΩ), gs))
    return unshaped(x), unshaped_nt_pullback
end

function ChainRulesCore.rrule(::typeof(unshaped), x::ShapedAsNT, vs::NamedTupleShape)
    gs = gradient_shape(valshape(x))
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), _snt_tangent(_normalize_tangent(ΔΩ), gs), NoTangent())
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


function ChainRulesCore.rrule(::Type{ShapedAsNT}, A::AbstractVector{<:Real}, vs::NamedTupleShape)
    shapedasnt_pullback(ΔΩ) = (NoTangent(), _flat_tangent(_write_snt_tangent!, _normalize_tangent(ΔΩ), vs, A), NoTangent())
    return ShapedAsNT(A, vs), shapedasnt_pullback
end


# Nested named tuples accessed directly via their accessors. The Zygote
# extension routes getindex and view with nested accessors here:

_accessor_tangent(write_nt!, data::AbstractVector{<:Real}, va::NamedTupleAccessor, dy::AbstractZero) = dy

function _accessor_tangent(write_nt!, data::AbstractVector{<:Real}, va::NamedTupleAccessor, dy)
    ddata = similar(data, promote_type(float(eltype(data)), _tangent_realtype(dy)))
    fill!(ddata, zero(eltype(ddata)))
    write_nt!(_flat_view(ddata, va), dy, va.shape)
    ddata
end

function ChainRulesCore.rrule(::typeof(ValueShapes.vs_getindex), data::AbstractVector{<:Real}, va::NamedTupleAccessor)
    vs_getindex_pullback(ΔΩ) = (NoTangent(), _accessor_tangent(_write_nt_tangent!, data, va, _normalize_tangent(ΔΩ)), NoTangent())
    return ValueShapes.vs_getindex(data, va), vs_getindex_pullback
end

function ChainRulesCore.rrule(::typeof(ValueShapes.vs_unsafe_view), data::AbstractVector{<:Real}, va::NamedTupleAccessor)
    vs_unsafe_view_pullback(ΔΩ) = (NoTangent(), _accessor_tangent(_write_snt_tangent!, data, va, _normalize_tangent(ΔΩ)), NoTangent())
    return ValueShapes.vs_unsafe_view(data, va), vs_unsafe_view_pullback
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


# Tangent storage with the layout of the array of flat vectors A:
_tangent_storage(A::AbstractArray{<:AbstractVector{<:Real}}, ::Type{T}) where {T<:Real} = map(x -> similar(x, T), A)
_tangent_storage(A::ArrayOfSimilarVectors{<:Real}, ::Type{T}) where {T<:Real} = ArrayOfSimilarVectors(similar(flatview(A), T))


struct GradShapedAsNTArrayProjector{VS<:NamedTupleShape} <: Function
    gradshape::VS
end

ChainRulesCore.ProjectTo(X::ShapedAsNTArray) = GradShapedAsNTArrayProjector(gradient_shape(elshape(X)))

function (project::GradShapedAsNTArrayProjector)(data::NamedTuple)
    _check_ntgs_tangent_compat(project.gradshape, get(data, :__internal_elshape, NoTangent()))
    _keep_zerolike(ShapedAsNTArray, _normalize_tangent(data.__internal_data), project.gradshape)
end

(project::GradShapedAsNTArrayProjector)(tangent::Tangent) = project(_backing(tangent))
(project::GradShapedAsNTArrayProjector)(tangent::AbstractThunk) = project(unthunk(tangent))
(project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(tangent::_AnySNTArray{names}) where names = tangent
(project::GradShapedAsNTArrayProjector)(tangent::_ZeroLike) = _az_tangent(tangent)

(project::GradShapedAsNTArrayProjector)(tangent::AbstractArray) =
    ShapedAsNTArray(_shapedasntarray_tangent(_normalize_tangent(tangent), project.gradshape), project.gradshape)


# Writes the normalized tangent A of one table column into the tangent storage data:
_write_snta_col!(data::AbstractArray{<:AbstractVector{<:Real}}, va::ValueAccessor, A::AbstractZero) =
    foreach(x -> fill!(_flat_view(x, va), zero(eltype(x))), data)

_write_snta_col!(data::AbstractArray{<:AbstractVector{<:Real}}, va::ValueAccessor, A::NamedTuple) =
    _write_snta_col!(data, va, A.__internal_data)

_write_snta_col!(data::AbstractArray{<:AbstractVector{<:Real}}, va::ValueAccessor, A::AbstractArray) =
    foreach((x, a) -> _write_tangent!(_flat_view(x, va), a, va.shape), data, A)

# Tangents of constant columns are irrelevant, whatever their representation:
_col_tangent(va::ValueAccessor, A) = A
_col_tangent(va::ValueAccessor{<:ConstValueShape}, A) = ZeroTangent()

function _tablecols_tangent(X::ShapedAsNTArray, dY::NamedTuple, gs::NamedTupleShape)
    T = promote_type(float(eltype(eltype(_data(X)))), _tangent_realtype(dY))
    dX_data = _tangent_storage(_data(X), T)
    map(values(_accessors(gs)), keys(_accessors(gs))) do va, p
        _write_snta_col!(dX_data, va, _col_tangent(va, get(dY, p, ZeroTangent())))
    end
    ShapedAsNTArray(dX_data, gs)
end

_tablecols_tangent(X::ShapedAsNTArray, dY::AbstractZero, gs::NamedTupleShape) = dY

function ChainRulesCore.rrule(::typeof(Tables.columns), X::ShapedAsNTArray)
    gs = gradient_shape(elshape(X))
    tablecols_pullback(ΔΩ) = (NoTangent(), _tablecols_tangent(X, _normalize_tangent(ΔΩ), gs))
    return Tables.columns(X), tablecols_pullback
end


_data_tangent(dY::AbstractArray{<:AbstractVector{<:Real}}, gs::NamedTupleShape) = ShapedAsNTArray(dY, gs)
_data_tangent(dY::AbstractZero, gs::NamedTupleShape) = dY

function ChainRulesCore.rrule(::typeof(_data), X::ShapedAsNTArray)
    gs = gradient_shape(elshape(X))
    _data_pullback(ΔΩ) = (NoTangent(), _data_tangent(_normalize_tangent(ΔΩ), gs))
    return _data(X), _data_pullback
end


# Array of flat tangents from a normalized tangent of a ShapedAsNTArray:
_shapedasntarray_tangent(dY::AbstractZero, vs::NamedTupleShape) = dY

_shapedasntarray_tangent(dY::NamedTuple, vs::NamedTupleShape) = dY.__internal_data

_shapedasntarray_tangent(dY::_AnySNTArray, vs::NamedTupleShape) = unshaped.(dY)

function _shapedasntarray_tangent(dY::AbstractArray, vs::NamedTupleShape)
    gs = gradient_shape(vs)
    T = promote_type(float(ValueShapes.default_unshaped_eltype(gs)), _tangent_realtype(dY))
    dX = ArrayOfSimilarVectors(Array{T}(undef, totalndof(gs), size(dY)...))
    foreach((dx, dy) -> _write_tangent!(dx, dy, gs), dX, dY)
    dX
end

function ChainRulesCore.rrule(::Type{ShapedAsNTArray}, A::AbstractArray{<:AbstractVector{<:Real}}, vs::NamedTupleShape)
    shapedasntarray_pullback(ΔΩ) = (NoTangent(), _shapedasntarray_tangent(_normalize_tangent(ΔΩ), vs), NoTangent())
    return ShapedAsNTArray(A, vs), shapedasntarray_pullback
end


end # module ValueShapesChainRulesCoreExt
