# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesMeasureBaseExt

using ValueShapes
using ValueShapes: AbstractValueShape, ScalarShape, ArrayShape, ConstValueShape, NamedTupleShape
using ValueShapes: NamedTupleDist, ConstValueDist, ReshapedDist, _distributions
using Distributions: Distribution
using InverseFunctions: inverse
import MeasureBase
using MeasureBase: AbstractMeasure, AsMeasure, Dirac, ProductMeasure, PowerMeasure, PushforwardMeasure, TransportFunction
using MeasureBase: DensityMeasure, WeightedMeasure, NoMSpaceElementSize
using MeasureBase: asmeasure, productmeasure, pushfwd, testvalue, marginals, mspace_flatsize

# The shape of a measure's variates, structural where the measure declares
# its variate layout, from a test value otherwise (real-valued shapes,
# whatever the element type of the test value):
ValueShapes.varshape(μ::AbstractMeasure) = _shape_of_value(testvalue(μ))
ValueShapes.varshape(μ::AsMeasure{<:Distribution}) = varshape(μ.obj)
ValueShapes.varshape(μ::Dirac) = ConstValueShape(μ.x)
ValueShapes.varshape(μ::ProductMeasure{<:NamedTuple}) = NamedTupleShape(map(varshape, marginals(μ)))
ValueShapes.varshape(μ::ProductMeasure{<:AbstractArray}) = _flat_shape(μ, mspace_flatsize(μ))
ValueShapes.varshape(μ::PowerMeasure) = _flat_shape(μ, mspace_flatsize(μ))
ValueShapes.varshape(μ::PushforwardMeasure{<:AbstractValueShape}) = μ.f
ValueShapes.varshape(μ::DensityMeasure) = varshape(μ.base)
ValueShapes.varshape(μ::WeightedMeasure) = varshape(μ.base)

_flat_shape(μ, sz::Tuple) = ArrayShape{Real}(map(Int, sz)...)
_flat_shape(μ, ::NoMSpaceElementSize) = _shape_of_value(testvalue(μ))

_shape_of_value(::Real) = ScalarShape{Real}()
_shape_of_value(x::AbstractArray{<:Real}) = ArrayShape{Real}(size(x)...)
_shape_of_value(x::NamedTuple) = NamedTupleShape(map(_shape_of_value, x))
_shape_of_value(x) = valshape(x)

# The measure over the flat vectors of its variates, and back:
ValueShapes.unshaped(μ::AbstractMeasure) = _unshaped(μ, varshape(μ))
_unshaped(μ, ::ArrayShape{<:Real,1}) = μ
_unshaped(μ, vs::AbstractValueShape) = pushfwd(inverse(vs), μ)

function ValueShapes.unshaped(μ::AbstractMeasure, vs::AbstractValueShape)
    varshape(μ) <= vs || throw(ArgumentError("Shape of the measure is not compatible with the given shape"))
    unshaped(μ)
end
ValueShapes.unshaped(μ::AbstractMeasure, vs::ConstValueShape) = invoke(unshaped, Tuple{AbstractMeasure,AbstractValueShape}, μ, vs)

(vs::AbstractValueShape)(μ::AbstractMeasure) = pushfwd(vs, μ)

# Transports know the shape of their results:
ValueShapes.resultshape(f::TransportFunction, ::AbstractValueShape) = varshape(f.ν)

# Named tuples of distributions are products of measures, constants are
# Dirac measures, reshaped distributions are pushforwards through their
# shape:
MeasureBase.asmeasure(d::NamedTupleDist) = productmeasure(map(asmeasure, _distributions(d)))
MeasureBase.asmeasure(d::ConstValueDist) = Dirac(d.value)
MeasureBase.asmeasure(d::ReshapedDist) = pushfwd(d.shape, asmeasure(d.dist))

end # module
