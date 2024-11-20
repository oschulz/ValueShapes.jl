# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesDistributionsChainRulesCoreExt

using ValueShapes

using Distributions: UnivariateDistribution

import ChainRulesCore
using ChainRulesCore: AbstractTangent, Tangent, AbstractZero, NoTangent, ZeroTangent
using ChainRulesCore: AbstractThunk, ProjectTo, unthunk, backing

_unshaped_uv_pullback(ΔΩ) = NoTangent(), only(unthunk(ΔΩ).v)

ChainRulesCore.rrule(::typeof(unshaped), d::UnivariateDistribution) = unshaped(d), _unshaped_uv_pullback

end # module ValueShapesDistributionsChainRulesCoreExt
