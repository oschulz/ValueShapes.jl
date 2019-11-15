# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


valshape(d::UnivariateDistribution) = ScalarShape{Real}()
valshape(d::MultivariateDistribution) = ArrayShape{Real}(size(d)...)
valshape(d::MatrixDistribution) = ArrayShape{Real}(size(d)...)
