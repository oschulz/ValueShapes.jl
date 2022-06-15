# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


#!!!!!!!!!!!!! ToDo: Remove varshape, vardof and unshaped for functions,
# untroduce retshape(f, ::AbstractValueShape...) instead.
# varshape becomes specific to distributions and measures


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

@inline _maybe_shaped(x::Any, vs::Nothing) = x
Base.@propagate_inbounds _maybe_shaped(x::Any, vs::AbstractValueShape) = stripscalar(vs(x))

@inline _maybe_unshaped(x::Any, vs::Nothing) = x
Base.@propagate_inbounds _maybe_unshaped(x::Any, vs::AbstractValueShape) = unshaped(x, vs)





# ToDo: Deprecate/remove:
struct FuncWithVarShape{F<:Function,VS<:AbstractValueShape} <: Function
    f::F
    varshape::VS
end

varshape(fws::FuncWithVarShape) = fws.varshape

Base.@propagate_inbounds (fws::FuncWithVarShape)(x::Any) = fws.f(x)

Base.@propagate_inbounds function (fws::FuncWithVarShape)(x::AbstractVector{<:Real})
    # ToDo: Check performance:
    x_shaped = varshape(fws)(x)
    fws.f(x_shaped)
end

# ToDo: Deprecate/remove:
import Base.>>
@inline >>(varshape::AbstractValueShape, f::Function) = FuncWithVarShape(f, varshape)
