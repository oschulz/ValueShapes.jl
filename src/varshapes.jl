# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).



const ValueShapeTuple{N} = NTuple{N,<:Integer} where {N}



struct VariableDataAccessor{N}
    shape::NTuple{N,Int}
    offset::Int
    len::Int

    VariableDataAccessor{N}(shape::NTuple{N,Int}, offset::Int) where {N} = new{N}(shape, offset, prod(shape))
end

VariableDataAccessor(shape::NTuple{N,Int}, offset::Int) where {N} = VariableDataAccessor{N}(shape, offset)

@inline function _view_range(idxs::AbstractUnitRange{<:Integer}, pa::VariableDataAccessor)
    from = first(idxs) + pa.offset
    to = from + pa.len - 1
    from:to
end


Base.@propagate_inbounds (pa::VariableDataAccessor{0})(parvalues::AbstractVector) =
    parvalues[first(_view_range(axes(parvalues, 1), pa))]

Base.@propagate_inbounds (pa::VariableDataAccessor{1})(parvalues::AbstractVector) =
    view(parvalues, _view_range(axes(parvalues, 1), pa))

Base.@propagate_inbounds (pa::VariableDataAccessor{N})(parvalues::AbstractVector) where N =
    reshape(view(parvalues, _view_range(axes(parvalues, 1), pa)), pa.shape...)


Base.@propagate_inbounds function (pa::VariableDataAccessor{0})(parvalues::AbstractVectorOfSimilarVectors)
    flat_parvalues = flatview(parvalues)
    idxs = _view_range(axes(flat_parvalues, 1), pa)
    view(flat_parvalues, first(idxs), :)
end

Base.@propagate_inbounds function (pa::VariableDataAccessor{1})(parvalues::AbstractVectorOfSimilarVectors)
    flat_parvalues = flatview(parvalues)
    idxs = _view_range(axes(flat_parvalues, 1), pa)
    fpview = view(flat_parvalues, idxs, :)
    VectorOfSimilarVectors(fpview)
end

Base.@propagate_inbounds function (pa::VariableDataAccessor{N})(parvalues::AbstractVectorOfSimilarVectors) where {N}
    flat_parvalues = flatview(parvalues)
    idxs = _view_range(axes(flat_parvalues, 1), pa)
    fpview = view(flat_parvalues, idxs, :)
    VectorOfSimilarArrays(reshape(fpview, pa.shape..., :))
end



@inline _paroffset_cumsum_impl(s, x, y, rest...) = (s, _paroffset_cumsum_impl(s+x, y, rest...)...)
@inline _paroffset_cumsum_impl(s,x) = (s,)
@inline _paroffset_cumsum_impl(s) = ()
@inline _paroffset_cumsum(x::Tuple) = _paroffset_cumsum_impl(0, x...)


"""
    VarShapes{N,AC}

Defines the shapes of an ordered set of variables (resp. parameters,
arguments, etc.). This forms the basis of viewing the content of all variables
in a dual way as a `NamedTuple` and as a flattened vectors.

Scalar values have shape `()`, array values have shape `(dim1, dim2, ...)`.

Constructors:

    VarShapes(name1 = shape1, ...)
    VarShapes(varshapes::NamedTuple)

e.g.

    varshapes = VarShapes(a = (2,3), b = (), c = (4,))

Use

    (varshapes::VarShapes)(data::AbstractVector)::NamedTuple

to get correctly named and shaped views into a vector containing the flattened
values of all variables. In return,

    Base.Vector{T}(::UndefInitializer, varshapes::VarShapes)

will create a suitable uninitialized vector to hold such flattened data for
a given set of variables.

When dealing with multiple vectors of flattened data,

    (varshapes::VarShapes)(
        data::ArrayOfArrays.AbstractVectorOfSimilarVectors
    )::NamedTuple

will transform a vector of vectors holding flattened data into a named tuple
of vectors, each tuple entry representing multiple values for a single
variable, implemented as views into `data`. The resulting tuple is suitable to
be wrapped into a `TypedTables.Table`, `DataFrames.DataFrame` or similar.

In return,

    ArraysOfArrays.VectorOfSimilarVectors{T}(varshapes::VarShapes)

will create a suitable vector (of length zero) of vectors to hold flattened
value data. The result will be a `VectorOfSimilarVectors` wrapped around a
2-dimensional `ElasticArray`. Internally all data is stored in a single
flat `Vector{T}`.

Example:

```julia
varshapes = VarShapes(a = (2,3), b = (), c = (4,))
data = VectorOfSimilarVectors{Int}(varshapes)
resize!(data, 10)
rand!(flatview(data), 0:99)
table = TypedTables.Table(varshapes(data))
fill!(table.b, 42)
all(x -> x == 42, view(flatview(data), 7, :))
```
"""
struct VarShapes{N,AC}
    _accessors::AC
    _n_pars_flattened::Int

    @inline function VarShapes(varshapes::NamedTuple{PN,<:NTuple{N,ValueShapeTuple}}) where {PN,N}
        labels = keys(varshapes)
        shapes = values(varshapes)
        lens = map(prod, shapes)
        offsets = _paroffset_cumsum(lens)
        accessors = map(VariableDataAccessor, shapes, offsets)
        lens = map(x -> x.len, accessors)
        n_flattened = sum(lens)
        named_accessors = NamedTuple{labels}(accessors)
        new{PN,typeof(named_accessors)}(named_accessors, n_flattened)
    end
end

export VarShapes

@inline VarShapes(;varshapes...) = VarShapes(varshapes.data)


@inline _accessors(x::VarShapes) = getfield(x, :_accessors)
@inline _n_pars_flattened(x::VarShapes) = getfield(x, :_n_pars_flattened)


@inline Base.keys(varshapes::VarShapes) = keys(_accessors(varshapes))

@inline Base.getproperty(varshapes::VarShapes, p::Symbol) = getfield(_accessors(varshapes), p)

@inline Base.propertynames(varshapes::VarShapes) = keys(varshapes)


Base.@propagate_inbounds function (varshapes::VarShapes)(parvalues::AbstractVector)
    accessors = _accessors(varshapes)
    map(pa -> pa(parvalues), accessors) # Not type-stable, investigate!
end

Base.@propagate_inbounds function (varshapes::VarShapes)(parvalues::AbstractVectorOfSimilarVectors)
    accessors = _accessors(varshapes)
    cols = map(pa -> pa(parvalues), accessors) # Not type-stable, investigate!
    cols
end


StatsBase.dof(varshapes::VarShapes) = _n_pars_flattened(varshapes)


Base.Vector{T}(::UndefInitializer, varshapes::VarShapes) where T =
    Vector{T}(undef, StatsBase.dof(varshapes))


ArraysOfArrays.VectorOfSimilarVectors{T}(varshapes::VarShapes) where T =
    VectorOfSimilarVectors(ElasticArray{T}(undef, StatsBase.dof(varshapes), 0))
