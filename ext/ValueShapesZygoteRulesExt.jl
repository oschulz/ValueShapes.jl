# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesZygoteRulesExt

using ValueShapes
using ValueShapes: ConstAccessor, NamedTupleAccessor, vs_getindex, vs_unsafe_view

import ZygoteRules


# Zygote has a generic `@adjoint getindex(x::AbstractArray, inds...)` and same for view that
# will result in overwriting va.shape.value with dy without these custom adjoints:
ZygoteRules.@adjoint function getindex(x::AbstractVector{<:Real}, va::ConstAccessor)
    getindex(x, va), dy -> nothing, nothing
end
ZygoteRules.@adjoint function view(x::AbstractVector{<:Real}, va::ConstAccessor)
    view(x, va), dy -> nothing, nothing
end


# The generic adjoints can't handle the structured values of nested named
# tuples either, use the ChainRulesCore rules of the ValueShapes accessor
# functions instead:
_tail_or_nothing(::Nothing) = nothing
_tail_or_nothing(x::Tuple) = Base.tail(x)

ZygoteRules.@adjoint function getindex(x::AbstractVector{<:Real}, va::NamedTupleAccessor)
    y, back = ZygoteRules._pullback(__context__, vs_getindex, x, va)
    y, dy -> _tail_or_nothing(back(dy))
end
ZygoteRules.@adjoint function view(x::AbstractVector{<:Real}, va::NamedTupleAccessor)
    checkbounds(x, va)
    y, back = ZygoteRules._pullback(__context__, vs_unsafe_view, x, va)
    y, dy -> _tail_or_nothing(back(dy))
end


end # module ValueShapesZygoteRulesExt
