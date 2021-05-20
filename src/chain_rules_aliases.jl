# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


@static if isdefined(ChainRulesCore, :AbstractTangent)
    const AbstractTangent = ChainRulesCore.AbstractTangent
    const NoTangent = ChainRulesCore.NoTangent
    const Tangent{P,T} = ChainRulesCore.Tangent{P,T}
    const ZeroTangent = ChainRulesCore.ZeroTangent
else
    const AbstractTangent = ChainRulesCore.AbstractDifferential
    const NoTangent = ChainRulesCore.DoesNotExist
    const Tangent{P,T} = ChainRulesCore.Composite{P,T}
    const ZeroTangent = ChainRulesCore.Zero
end
