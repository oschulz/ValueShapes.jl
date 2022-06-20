# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

function Base.show(io::IO, d::NamedTupleDist)
    print(io, Base.typename(typeof(d)).name, "{")
    show(io, propertynames(d))
    print(io, "}(…)")
end



# ToDo: Find more generic way of handling this:
getdof(d::NamedTupleDist) = sum(map(getdof, values(d)))


_flat_ntd_elshape(d::Distribution) = ArrayShape{Real}(getdof(d))

function _flat_ntd_accessors(d::NamedTupleDist{names,DT,AT,VT}) where {names,DT,AT,VT}
    shapes = map(_flat_ntd_elshape, values(d))
    vs = NamedTupleShape(VT, NamedTuple{names}(shapes))
    values(vs)
end


function _flat_ntdistelem_to_stdmv(trg::StandardDist{<:Any,1}, sd::Distribution, x_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor)
    td = view(trg, ValueShapes.view_idxs(Base.OneTo(length(trg)), trg_acc))
    sv = trg_acc(x_unshaped)
    vartransform(td, unshaped(sd), sv)
end

function _flat_ntdistelem_to_stdmv(trg::StandardDist{<:Any,1}, sd::ConstValueDist, x_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor)
    Bool[]
end

function MeasureBase.vartransform(trg::StandardDist{<:Any,1}, src::ValueShapes.UnshapedNTD, x::AbstractVector{<:Real})
    @argcheck length(src) == length(eachindex(x))
    trg_accessors = _flat_ntd_accessors(src.shaped)
    rs = map((acc, sd) -> _flat_ntdistelem_to_stdmv(trg, sd, x, acc), trg_accessors, values(src.shaped))
    vcat(rs...)
end

function MeasureBase.vartransform(trg::StandardDist{<:Any,1}, src::NamedTupleDist, x::Union{NamedTuple,ShapedAsNT})
    x_unshaped = unshaped(x, varshape(src))
    vartransform(trg, unshaped(src), x_unshaped)
end


function _stdmv_to_flat_ntdistelem(td::Distribution, src::StandardDist{<:Any,1}, x::AbstractVector{<:Real}, src_acc::ValueAccessor)
    sd = view(src, ValueShapes.view_idxs(Base.OneTo(length(src)), src_acc))
    sv = src_acc(x)
    vartransform(unshaped(td), sd, sv)
end

function _stdmv_to_flat_ntdistelem(td::ConstValueDist, src::StandardDist{<:Any,1}, x::AbstractVector{<:Real}, src_acc::ValueAccessor)
    Bool[]
end

function MeasureBase.vartransform(trg::ValueShapes.UnshapedNTD, src::StandardDist{<:Any,1}, x::AbstractVector{<:Real})
    @argcheck length(src) == length(eachindex(x))
    src_accessors = _flat_ntd_accessors(trg.shaped)
    rs = map((acc, td) -> _stdmv_to_flat_ntdistelem(td, src, x, acc), src_accessors, values(trg.shaped))
    vcat(rs...)
end

function MeasureBase.vartransform(trg::NamedTupleDist, src::StandardDist{<:Any,1}, x::AbstractVector{<:Real})
    unshaped_result = vartransform(unshaped(trg), src, x)
    varshape(trg)(unshaped_result)
end
