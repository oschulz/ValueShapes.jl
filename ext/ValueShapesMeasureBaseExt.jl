# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesMeasureBaseExt

using ValueShapes
using ValueShapes: AbstractValueShape, ScalarShape, ArrayShape, ConstValueShape, NamedTupleShape
using ValueShapes: NamedTupleDist, ConstValueDist, ReshapedDist, _distributions
using Distributions: Distribution
using InverseFunctions: inverse
using Random: Xoshiro
import MeasureBase
using MeasureBase: AbstractMeasure, AsMeasure, Dirac, ProductMeasure, PowerMeasure, PushforwardMeasure, TransportFunction
using MeasureBase: DensityMeasure, WeightedMeasure, NoMSpaceElementSize, Bind
using MeasureBase: asmeasure, productmeasure, pushfwd, testvalue, marginals, mspace_flatsize, boundmeasure, bindkernel

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

# Binds have the shapes of their primary and secondary variates, combined
# like the variates. They only have a fixed shape if the shape of the
# secondary measure doesn't depend on the primary variate, this is checked
# at the test value and at a random draw of the primary measure:
function ValueShapes.varshape(μ::Bind)
    _combined_shape(_bind_combinator(μ), varshape(boundmeasure(μ)), _secondary_shape(μ))
end

function _secondary_shape(μ::Bind)
    α = boundmeasure(μ)
    vs = varshape(_secondary_measure(μ, testvalue(α)))
    vs == varshape(_secondary_measure(μ, rand(Xoshiro(0), α))) ||
        throw(ArgumentError("Bind has no fixed variate shape, the shape of its secondary measure depends on the primary variate"))
    return vs
end

# MeasureBase has no public accessors for the parts of a bind kernel:
_bind_combinator(μ::Bind) = bindkernel(μ).f_c
_secondary_measure(μ::Bind, a) = asmeasure(bindkernel(μ).f_β(a))

_combined_shape(::typeof(MeasureBase.secondarg), ::AbstractValueShape, vs_b::AbstractValueShape) = vs_b
_combined_shape(::typeof(merge), vs_a::NamedTupleShape, vs_b::NamedTupleShape) = merge(vs_a, vs_b)
_combined_shape(::typeof(vcat), vs_a::AbstractValueShape, vs_b::AbstractValueShape) =
    ArrayShape{Real}(_vcat_length(vs_a) + _vcat_length(vs_b))
function _combined_shape(f_c, ::AbstractValueShape, ::AbstractValueShape)
    throw(ArgumentError("Variate shapes of values combined via $f_c are not supported"))
end

_vcat_length(::ScalarShape{<:Real}) = 1
_vcat_length(vs::ArrayShape{<:Real,1}) = totalndof(vs)

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
