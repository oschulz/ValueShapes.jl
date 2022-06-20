# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    UnknownVarShape(x)

`varshape(x) == UnknownVarShape(x)` indicates that variate shape
information is not available for `x`
"""
struct UnknownVarShape{T}
    x::T
end

export UnknownVarShape


"""
    varshape(x::Any)::AbstractValueShape

Get the shape of the variates of `x`. `x` may be a measure,
distributions, random variable, sampler or similar.

Defaults to `UnknownVarShape(x)`.
"""
varshape(x::Any) = UnknownVarShape(x)
export varshape
