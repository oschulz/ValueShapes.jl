# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    varshape(d::Distributions.Distribution)::AbstractValueShape

Get the value shape of the variates of distribution `d`.
"""
varshape(d::Distribution{Univariate}) = ScalarShape{Real}()
varshape(d::Distribution{<:ArrayLikeVariate}) = ArrayShape{Real}(size(d)...)



const UvDistAsMv{D<:Distribution{Univariate}} = Product{<:Any,D,Fill{D,1,Tuple{Base.OneTo{StaticInt{1}}}}}

"""
    unshaped(d::Distributions.Distribution)

Turns `d` into a `Distributions.Distribution{Multivariate}` based on
`varshape(d)`.
"""
function unshaped(d::UnivariateDistribution)
    # ToDo: Replace with `reshape(d, 1)` when result of `reshape(::UnivariateDistribution, 1)`
    # becomes fully functional in Distributions:
    Distributions.Product(Fill(d, static(1)))
end

@inline MeasureBase.transport_origin(ν::UvDistAsMv) = only(ν.v)
@inline MeasureBase.from_origin(ν::UvDistAsMv, x) = Fill(x, static(1))
@inline MeasureBase.to_origin(ν::UvDistAsMv, y) = only(y)


unshaped(d::Distribution{Multivariate}) = d

@static if isdefined(Distributions, :ReshapedDistribution)
    unshaped(d::Distribution{<:ArrayLikeVariate}) = reshape(d, length(d))
else
    unshaped(d::MatrixReshaped) = d.d
end


"""
    struct StructVariate{T}

Variate kind for structured type of type `T`.
"""
struct StructVariate{T} <: VariateForm end
