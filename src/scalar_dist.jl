# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


# ToDo: Replace with custom UnshapedUvDist:
unshaped(d::UnivariateDistribution) = Distributions.Product(Fill(d))
