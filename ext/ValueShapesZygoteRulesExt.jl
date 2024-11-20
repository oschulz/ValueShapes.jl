# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesZygoteRulesExt

using ValueShapes: ConstAccessor

import ZygoteRules


# Zygote has a generic `@adjoint getindex(x::AbstractArray, inds...)` and same for view that
# will result in overwriting va.shape.value with dy without these custom adjoints:
ZygoteRules.@adjoint function getindex(x::AbstractVector{<:Real}, va::ConstAccessor)
    getindex(x, va), dy -> nothing, nothing
end
ZygoteRules.@adjoint function view(x::AbstractVector{<:Real}, va::ConstAccessor)
    view(x, va), dy -> nothing, nothing
end


end # module ValueShapesZygoteRulesExt
