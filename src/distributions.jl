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


Base.rand(s::Sampleable{<:StructVariate}, n::Int) = rand(Random.GLOBAL_RNG, s, n)

Base.rand(s::Sampleable{<:StructVariate}, dims::Dims) = rand(Random.GLOBAL_RNG, s, dims)

Base.rand(s::Sampleable{<:StructVariate}, dims::Dims, A::AbstractArray) = rand(Random.GLOBAL_RNG, s, dims)

Base.rand(rng::AbstractRNG, s::Sampleable{<:StructVariate}, n::Int) = rand(rng, s, Dims((n,)))

function Base.rand(rng::AbstractRNG, s::Sampleable{<:StructVariate}, dims::Dims)
    broadcast(x -> rand(rng, s), Fill(nothing, dims...))
end

function Random.rand!(rng::AbstractRNG, s::Sampleable{<:StructVariate}, A::AbstractArray)
    broadcast!(x -> rand(rng, s), A, A)
end



unshaped(d::MultivariateDistribution) = d
