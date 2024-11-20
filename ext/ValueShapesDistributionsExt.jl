# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesDistributionsExt

using ValueShapes

using Distributions

import ValueShapes: varshape, unshaped


"""
    varshape(d::Distributions.Distribution)::AbstractValueShape

Get the value shape of the variates of distribution `d`.
"""
function varshape end
export varshape

varshape(d::Distribution{Univariate}) = ScalarShape{Real}()
@static if isdefined(Distributions, :ArrayLikeVariate)
    varshape(d::Distribution{<:ArrayLikeVariate}) = ArrayShape{Real}(size(d)...)
else
    varshape(d::Distribution{Multivariate}) = ArrayShape{Real}(size(d)...)
    varshape(d::Distribution{Matrixvariate}) = ArrayShape{Real}(size(d)...)
end


"""
    unshaped(d::Distributions.Distribution)

Turns `d` into a `Distributions.Distribution{Multivariate}` based on
`varshape(d)`.
"""
function unshaped(d::UnivariateDistribution)
    # ToDo: Replace with `reshape(d, 1)` when result of `reshape(::UnivariateDistribution, 1)`
    # becomes fully functional in Distributions:
    product_distribution(Fill(d, 1))
end

unshaped(d::Distribution{Multivariate}) = d

@static if isdefined(Distributions, :ReshapedDistribution)
    unshaped(d::Distribution{<:ArrayLikeVariate}) = reshape(d, length(d))
else
    unshaped(d::MatrixReshaped) = d.d
end



end # module ValueShapesDistributionsExt
