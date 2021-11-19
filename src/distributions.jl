# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    varshape(d::Distributions.Distribution)::AbstractValueShape

Get the value shape of the variates of distribution `d`.
"""
varshape(d::Distribution{Univariate}) = ScalarShape{Real}()
varshape(d::Distribution{Multivariate}) = ArrayShape{Real}(size(d)...)
varshape(d::Distribution{Matrixvariate}) = ArrayShape{Real}(size(d)...)

@deprecate valshape(d::Distribution) varshape(d)


"""
    vardof(d::Distributions.Distribution)

Get the number of degrees of freedom of the variates of distribution `d`.
Equivalent to `totalndof(varshape(d))`.
"""
vardof(d::Distribution) = totalndof(varshape(d))


"""
    unshaped(d::Distributions.Distribution)

Turns `d` into a `Distributions.Distribution{Multivariate}` based on
`varshape(d)`.
"""
unshaped(d::Distribution{Multivariate}) = d
unshaped(d::MatrixReshaped) = d.d



const PlainVariate = Union{Univariate,Multivariate,Matrixvariate}


struct StructVariate{T} <: VariateForm end


const NamedTupleVariate{names} = StructVariate{NamedTuple{names}}  # ToDo: Use StructVariate{<:NamedTuple{names}} instead?


Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}) where names = stripscalar(rand(rng, d, ()))

function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, dims::Tuple{}) where names
    view(rand(rng, d, 1), 1)
end

function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, dims::Dims) where names
    shape = varshape(d)
    X_flat = Array{default_unshaped_eltype(shape)}(undef, totalndof(varshape(d)), dims...)
    X = ArrayOfSimilarVectors(X_flat)
    rand!(rng, unshaped(d), X)
    shape.(X)
end


function Random.rand!(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    valshape(x) >= varshape(d) || throw(ArgumentError("Shapes of variate and value are not compatible"))
    rand!(rng, unshaped(d), unshaped(x))
    x
end


function _aov_rand_impl!(rng::AbstractRNG, d::Distribution{Multivariate}, X::ArrayOfSimilarVectors{<:Real})
    rand!(rng, unshaped(d), flatview(X))
end

# Workaround for current limitations of ArraysOfArrays.unshaped for standard arrays of vectors
function _aov_rand_impl!(rng::AbstractRNG, d::Distribution{Multivariate}, X::AbstractArray{<:AbstractVector{<:Real}})
    rand!.(Ref(rng), Ref(unshaped(d)), X)
end

function Random.rand!(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, X::ShapedAsNTArray{<:NamedTuple{names}}) where names
    elshape(X) >= varshape(d) || throw(ArgumentError("Shapes of variate and value are not compatible"))
    _aov_rand_impl!(rng, unshaped(d), unshaped.(X))
    X
end


function Distributions.logpdf(d::Distribution{NamedTupleVariate{names}}, x::NamedTuple{names}) where names
    logpdf(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.logpdf(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    @argcheck valshape(x) <= varshape(d)
    logpdf(unshaped(d), unshaped(x))
end

function Distributions.logpdf(d::Distribution{NamedTupleVariate{names}}, x::AbstractArray{<:NamedTuple{names},0}) where names
    logpdf(unshaped(d), unshaped(x, varshape(d)))
end


function Distributions.pdf(d::Distribution{NamedTupleVariate{names}}, x::NamedTuple{names}) where names
    pdf(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.pdf(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    @argcheck valshape(x) <= varshape(d)
    pdf(unshaped(d), unshaped(x))
end

function Distributions.pdf(d::Distribution{NamedTupleVariate{names}}, x::AbstractArray{<:NamedTuple{names},0}) where names
    pdf(unshaped(d), unshaped(x, varshape(d)))
end


function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, x::NamedTuple{names}) where names
    insupport(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    @argcheck valshape(x) <= varshape(d)
    insupport(unshaped(d), unshaped(x))
end

function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, x::AbstractArray{<:NamedTuple{names},0}) where names
    insupport(unshaped(d), unshaped(x, varshape(d)))
end

function Distributions.insupport(d::Distribution{NamedTupleVariate{names}}, X::AbstractArray{<:NamedTuple{names},N}) where {N,names}
    Distributions.insupport!(BitArray(undef, size(X)), d, X)
end

function Distributions.insupport!(r::AbstractArray{Bool,N}, d::Distribution{NamedTupleVariate{names}}, X::AbstractArray{<:NamedTuple{names},N}) where {N,names}
    r .= insupport.(Ref(d), X)
end
