# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

const MvStdMeasure = PowerMeasure{<:StdMeasure,<:NTuple{1,Base.OneTo}}


function resultshape(f::MeasureBase.VarTransformation, vs::AbstractValueShape)
    @argcheck vs <= varshape(f.μ)
    return varshape(trafo.ν)
end


function Base.Broadcast.broadcasted(
    trafo::VarTransformation,
    v_src::Union{ArrayOfSimilarVectors{<:Real},ShapedAsNTArray}
)
    return broadcast_trafo(trafo, v_src)
end

