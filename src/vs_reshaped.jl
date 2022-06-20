# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


MeasureBase.effndof(d::ValueShapes.ReshapedDist) = MeasureBase.effndof(unshaped(d))

MeasureBase.vartransform_origin(d::ValueShapes.ReshapedDist) = unshaped(d)

MeasureBase.to_origin(src::ValueShapes.ReshapedDist, x) = unshaped(x, varshape(src))

MeasureBase.from_origin(trg::ValueShapes.ReshapedDist, x) = varshape(trg)(x)
