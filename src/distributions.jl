# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    varshape(d::Distributions.Distribution)::AbstractValueShape

Get the value shape of the variates of distribution `d`.
"""
varshape(d::UnivariateDistribution) = ScalarShape{Real}()
varshape(d::MultivariateDistribution) = ArrayShape{Real}(size(d)...)
varshape(d::MatrixDistribution) = ArrayShape{Real}(size(d)...)

@deprecate valshape(d::Distribution) varshape(d)


"""
    vardof(d::Distributions.Distribution)

Get the number of degrees of freedom of the variates of distribution `d`.
Equivalent to `totalndof(varshape(d))`.
"""
vardof(d::Distribution) = totalndof(varshape(d))
