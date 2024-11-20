# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesAdaptExt

using ValueShapes
import Adapt

function Adapt.adapt_structure(to, x::ShapedAsNTArray)
    ShapedAsNTArray(Adapt.adapt(to, _data(x)), _elshape(x))
end

function Adapt.adapt_structure(to, x::ShapedAsNT)
    ShapedAsNT(Adapt.adapt(to, _data(x)), _valshape(x))
end

end # module ValueShapesAdaptExt
