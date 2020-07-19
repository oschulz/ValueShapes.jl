# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


unshaped(d::UnivariateDistribtuion) = Distributions.Product(Fill(d))
