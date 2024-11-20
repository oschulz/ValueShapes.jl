# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesInverseFunctionsExt

using ValueShapes: AbstractValueShape, _InvValueShape

import InverseFunctions


InverseFunctions.inverse(vs::AbstractValueShape) = Base.Fix2(unshaped, vs)
InverseFunctions.inverse(inv_vs::_InvValueShape) = inv_vs.x

end # module ValueShapesInverseFunctionsExt
