# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

"""
    UnknownReturnShape{F}(argshape)

Indicates that that shape of the return value for functions of type `F` and
arguments of shape `argshape` cannot be determined.

See [`retshape`](@ref).
"""
struct UnknownReturnShape{F,S}
    argshape::S
end

UnknownReturnShape{F}(argshape::S) where {F,S<:AbstractValueShape} = UnknownReturnShape{F,S}(argshape)
UnknownReturnShape{F}(argshape::UnknownReturnShape) where {F} = argshape


"""
    retshape(f::F, argshape::AbstractValueShape)::AbstractValueShape

Compute the value shape of `y = f(x)` for values `x` with shape `argshape`.

Defaults to `UnknownReturnShape{F}(shape_x)`.
"""
@inline retshape(::F, argshape::AbstractValueShape) where {F} = UnknownReturnShape{F}(argshape)
export retshape

@inline retshape(::typeof(identity), argshape::AbstractValueShape) = argshape

@inline retshape(f::ComposedFunction, argshape::AbstractValueShape) = retshape(f.outer, retshape(f.inner, argshape))
