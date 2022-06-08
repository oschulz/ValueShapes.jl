# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


@inline _adignore_call(f) = f()
@inline _adignore_call_pullback(@nospecialize ΔΩ) = (NoTangent(), NoTangent())
ChainRulesCore.rrule(::typeof(_adignore_call), f) = _adignore_call(f), _adignore_call_pullback

macro _adignore(expr)
    :(_adignore_call(() -> $(esc(expr))))
end


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


@inline _keep_zerolike(::Type{T}, x, xs...) where T = T(x, xs...)
@inline _keep_zerolike(::Type{T}, x::_ZeroLike, xs...) where T = _az_tangent(x)
@inline _keep_zerolike(f::F, x, xs...) where F = f(x, xs...)
@inline _keep_zerolike(f::F, x::_ZeroLike, xs...) where F = _az_tangent(x)
