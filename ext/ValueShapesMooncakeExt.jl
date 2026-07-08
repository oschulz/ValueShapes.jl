# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesMooncakeExt

using ValueShapes: _adignore_call

import Mooncake

Mooncake.@zero_adjoint Mooncake.MinimalCtx Tuple{typeof(_adignore_call),Any}

end # module ValueShapesMooncakeExt
