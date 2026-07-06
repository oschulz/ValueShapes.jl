# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesAdaptExt

using ValueShapes
using ValueShapes: _data, _valshape, _elshape

import Adapt

function Adapt.adapt_structure(to, x::ShapedAsNT)
    ShapedAsNT(Adapt.adapt(to, _data(x)), _valshape(x))
end

function Adapt.adapt_structure(to, x::ShapedAsNTArray)
    ShapedAsNTArray(Adapt.adapt(to, _data(x)), _elshape(x))
end

end # module ValueShapesAdaptExt
