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



const PlainVariate = Union{Univariate,Multivariate,Matrixvariate}


struct StructVariate{T} <: VariateForm end

const NamedTupleVariate{names} = StructVariate{NamedTuple{names}}


rand(s::Sampleable{<:StructVariate}, n::Int) = rand(Random.GLOBAL_RNG, s, n)

rand(s::Sampleable{<:StructVariate}, dims::Dims) = rand(Random.GLOBAL_RNG, s, dims)

rand(s::Sampleable{<:StructVariate}, dims::Dims, A::AbstractArray) = rand(Random.GLOBAL_RNG, s, dims)

rand(rng::AbstractRNG, s::Sampleable{<:StructVariate}, n::Int) = rand(rng, s, Dims((n,)))

function rand(rng::AbstractRNG, s::Sampleable{<:StructVariate}, dims::Dims)
    broadcast(x -> rng(rng, s), Fill(nothing, dims...))
end

function rand!(rng::AbstractRNG, s::Sampleable{<:StructVariate}, A::AbstractArray)
    broadcast!(x -> rng(rng, s), A, A)
end
