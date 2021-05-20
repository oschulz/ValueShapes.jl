# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    varshape(f::Function)::AbstractValueShape

Get the value shape of the input/argument of an unary function `f`. `f`
should support call syntax

    f(x::T)

with `valshape(x) == varshape(f)` as well all

    f(x::AbstractVector{<:Real})

with `length(eachindex(x)) == vardof(f)` (see [`vardof`](@ref)).
"""
function varshape end
export varshape


"""
    vardof(f::Function)::Integer

Get the number of degrees of freedom of the input/argument of f.

Equivalent to `totalndof(varshape(f))` (see [`varshape`](@ref)).
"""
function vardof end
export vardof

vardof(f::Function) = totalndof(varshape(f))



@inline _maybe_shaped(x::Any, vs::Nothing) = x
Base.@propagate_inbounds _maybe_shaped(x::Any, vs::AbstractValueShape) = stripscalar(vs(x))

@inline _maybe_unshaped(x::Any, vs::Nothing) = x
Base.@propagate_inbounds _maybe_unshaped(x::Any, vs::AbstractValueShape) = unshaped(x, vs)


struct UnshapedFunction{F<:Function,IS<:Union{Nothing,AbstractValueShape},OS<:Union{Nothing,AbstractValueShape}} <: Function
    orig_f::F
    orig_varshape::IS
    orig_valshape::OS
end

"""
    unshaped(
        f::Function,
        orig_varshape::Union{Nothing,AbstractValueShape},
        orig_valshape::Union{Nothing,AbstractValueShape} = nothing
    )

Return a function that
    * Shapes it's input from a flat vector of `Real` using `orig_varshape`, if not `nothing`.
    * Calls `f` with the (optionally) unshaped input.
    * Unshapes the result to a flat vector of `Real` using `orig_valshape`, if not `nothing`.
"""
function unshaped(f::Function, orig_varshape::Union{Nothing,AbstractValueShape}, orig_valshape::Union{Nothing,AbstractValueShape} = nothing)
    UnshapedFunction(f, orig_varshape, orig_valshape)
end

varshape(uf::UnshapedFunction) = uf.varshape

Base.@propagate_inbounds function (uf::UnshapedFunction)(x)
    orig_x =_maybe_shaped(x, uf.orig_varshape)
    orig_y = uf.orig_f(orig_x) 
    _maybe_unshaped(orig_y, uf.orig_valshape)
end





# ToDo: Deprecate/remove:
struct FuncWithVarShape{F<:Function,VS<:AbstractValueShape} <: Function
    f::F
    varshape::VS
end

varshape(fws::FuncWithVarShape) = fws.varshape

Base.@propagate_inbounds (fws::FuncWithVarShape)(x::Any) = fws.f(x)

Base.@propagate_inbounds function (fws::FuncWithVarShape)(x::AbstractVector{<:Real})
    # ToDo: Check performance:
    x_shaped = stripscalar(varshape(fws)(x))
    fws.f(x_shaped)
end

# ToDo: Deprecate/remove:
import Base.>>
@inline >>(varshape::AbstractValueShape, f::Function) = FuncWithVarShape(f, varshape)
