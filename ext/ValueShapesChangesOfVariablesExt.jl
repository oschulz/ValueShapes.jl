# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

module ValueShapesChangesOfVariablesExt

using ValueShapes: AbstractValueShape, _InvValueShape

import InverseFunctions


function ChangesOfVariables.with_logabsdet_jacobian(bc_vs::_BroadcastValueShape, ao_flat_x)
    ao_x = bc_vs(ao_flat_x)
    ao_x, zero(float(realnumtype(typeof(ao_flat_x))))
end

function ChangesOfVariables.with_logabsdet_jacobian(vs::AbstractValueShape, flat_x)
    x = vs(flat_x)
    x, zero(float(eltype(flat_x)))
end

function ChangesOfVariables.with_logabsdet_jacobian(inv_vs::_InvValueShape, x)
    flat_x = inv_vs(x)
    flat_x, zero(float(eltype(flat_x)))
end

function ChangesOfVariables.with_logabsdet_jacobian(bc_inv_vs::Union{_BroadcastInvValueShape,_BroadcastUnshaped}, ao_x)
    ao_flat_x = bc_inv_vs(ao_x)
    ao_flat_x, zero(float(realnumtype(typeof(ao_flat_x))))
end


end # module ValueShapesChangesOfVariablesExt
