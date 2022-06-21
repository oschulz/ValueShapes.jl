# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

const UvStdMeasure = MeasureBase.StdMeasure
const MvStdMeasure{D<:MeasureBase.StdMeasure} = PowerMeasure{D,<:NTuple{1,Base.OneTo}}


@inline varshape(μ::MeasureBase.StdMeasure) = ScalarShape{Real}()

@inline _power_varshape(::ScalarShape{T}, dims::Dims) where T = ArrayShape{T}(dims...)
function _power_varshape(vs::AbstractValueShape, ::Dims)
    throw(ArgumentError("Can't express shape of array with element shape $vs (yet)"))
end

@inline varshape(μ::PowerMeasure) = _power_varshape(varshape(μ.parent), map(length, μ.axes))


unshaped(μ::DistributionMeasure) = DistributionMeasure(unshaped(μ.d))
varshape(μ::DistributionMeasure) = varshape(μ.d)


function retshape(f::MeasureBase.TransportFunction, vs::AbstractValueShape)
    @argcheck vs <= varshape(f.μ)
    return varshape(trafo.ν)
end




#!!!!!!!!!!!!!!!!!
#=
function Base.Broadcast.broadcasted(
    trafo::TransportFunction,
    v_src::Union{ArrayOfSimilarVectors{<:Real},ShapedAsNTArray}
)
    return broadcast_trafo(trafo, v_src)
end
=#