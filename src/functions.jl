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



struct FuncWithVarShape{F<:Function,VS<:AbstractValueShape} <: Function
    f::F
    varshape::VS
end

varshape(fws::FuncWithVarShape) = fws.varshape

Base.@propagate_inbounds (fws::FuncWithVarShape)(x::Any) = fws.f(x)

Base.@propagate_inbounds (fws::FuncWithVarShape)(x::AbstractVector{<:Real}) =
    fws.f(varshape(fws)(x))


import Base.>>
@inline >>(varshape::AbstractValueShape, f::Function) = FuncWithVarShape(f, varshape)
