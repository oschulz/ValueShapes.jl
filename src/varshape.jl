# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    MissingVarShape(x)

`varshape(x) == MissingVarShape(x)` indicates that variate shape
information is not available for `x`
"""
struct MissingVarShape{T}
    x::T
end

export MissingVarShape


"""
    varshape(md::Any)::AbstractValueShape

Get the total number of degrees of freedom variates of the distribution-
resp. measure-like object `md`.

Defaults to `varshape(x::Any) == MissingVarShape(x)`.
"""
varshape(x::Any) = MissingVarShape(x)
export varshape



"""
    MissingTotalNDOF(x)

`varshape(x) == MissingTotalNDOF(x)` indicates that the total number of
degrees of freedom is not available for `x`
"""
struct MissingTotalNDOF{T}
    x::T
end

export MissingTotalNDOF


"""
    vardof(md::Any)::Integer

Get the total number of degrees of freedom variates of the distribution-
resp. measure-like object `md`.

Defaults to `totalndof(varshape(md))` (see [`varshape`](@ref)) and
`MissingTotalNDOF(x)` if `varshape(md) <: MissingVarShape`.
"""
function vardof end
export vardof

function vardof(md::Any)
    vs = varshape(md)
    vs <: MissingVarShape ? MissingTotalNDOF(x) : totalndof(vs)
end
