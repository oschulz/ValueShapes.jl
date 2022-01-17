# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    varshape(d::Distributions.Distribution)::AbstractValueShape

Get the value shape of the variates of distribution `d`.
"""
varshape(d::Distribution{Univariate}) = ScalarShape{Real}()
@static if isdefined(Distributions, :ArrayLikeVariate)
    varshape(d::Distribution{<:ArrayLikeVariate}) = ArrayShape{Real}(size(d)...)
else
    varshape(d::Distribution{Multivariate}) = ArrayShape{Real}(size(d)...)
    varshape(d::Distribution{Matrixvariate}) = ArrayShape{Real}(size(d)...)
end

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
function unshaped(d::UnivariateDistribution)
    # ToDo: Replace with `reshape(d, 1)` when result of `reshape(::UnivariateDistribution, 1)`
    # becomes fully functional in Distributions:
    Distributions.Product(Fill(d, 1))
end

unshaped(d::Distribution{Multivariate}) = d

@static if isdefined(Distributions, :ReshapedDistribution)
    unshaped(d::Distribution{<:ArrayLikeVariate}) = reshape(d, length(d))
else
    unshaped(d::MatrixReshaped) = d.d
end


@static if isdefined(Distributions, :ArrayLikeVariate)
    const PlainVariate = ArrayLikeVariate
else
    const PlainVariate = Union{Univariate,Multivariate,Matrixvariate}
end

struct StructVariate{T} <: VariateForm end


const NamedTupleVariate{names} = StructVariate{NamedTuple{names}}  # ToDo: Use StructVariate{<:NamedTuple{names}} instead?


function _rand_flat_impl(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}) where names
    shape = varshape(d)
    X = Vector{default_unshaped_eltype(shape)}(undef, totalndof(varshape(d)))
    (shape, rand!(rng, unshaped(d), X))
end

function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}) where names
    shape, X = _rand_flat_impl(rng, d)
    shape(X)
end

function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, dims::Tuple{}) where names
    shape, X = _rand_flat_impl(rng, d)
    shape.(Fill(X))
end


function Random.rand(rng::AbstractRNG, d::Distribution{NamedTupleVariate{names}}, dims::Dims) where names
    shape = varshape(d)
    X_flat = Array{default_unshaped_eltype(shape)}(undef, totalndof(varshape(d)), dims...)
    X = ArrayOfSimilarVectors(X_flat)
    rand!(rng, unshaped(d), X)
    shape.(X)
end


function Random.rand!(d::Distribution{NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    rand!(Random.default_rng(), d, x)
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

function Random.rand!(rng::AbstractRNG, d::Distribution{<:NamedTupleVariate}, X::ShapedAsNTArray)
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
