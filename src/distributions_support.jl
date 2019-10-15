# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


valshape(d::Distributions.UnivariateDistribution) = ScalarShape{Real}()
valshape(d::Distributions.MultivariateDistribution) = ArrayShape{Real}(size(d)...)
valshape(d::Distributions.MatrixDistribution) = ArrayShape{Real}(size(d)...)
