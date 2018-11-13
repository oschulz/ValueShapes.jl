# This file is a part of ParameterShapes.jl, licensed under the MIT License (MIT).



const ParamShapeTuple{N} = NTuple{N,<:Integer} where {N}



struct ParamDataAccessor{N}
    shape::NTuple{N,Int}
    offset::Int
    len::Int

    ParamDataAccessor{N}(shape::NTuple{N,Int}, offset::Int) where {N} = new{N}(shape, offset, prod(shape))
end

ParamDataAccessor(shape::NTuple{N,Int}, offset::Int) where {N} = ParamDataAccessor{N}(shape, offset)

@inline function _view_range(idxs::AbstractUnitRange{<:Integer}, pa::ParamDataAccessor)
    from = first(idxs) + pa.offset
    to = from + pa.len - 1
    from:to
end


Base.@propagate_inbounds (pa::ParamDataAccessor{0})(parvalues::AbstractVector) =
    parvalues[first(_view_range(axes(parvalues, 1), pa))]

Base.@propagate_inbounds (pa::ParamDataAccessor{1})(parvalues::AbstractVector) =
    view(parvalues, _view_range(axes(parvalues, 1), pa))

Base.@propagate_inbounds (pa::ParamDataAccessor{N})(parvalues::AbstractVector) where N =
    reshape(view(parvalues, _view_range(axes(parvalues, 1), pa)), pa.shape...)


Base.@propagate_inbounds function (pa::ParamDataAccessor{0})(parvalues::AbstractVectorOfSimilarVectors)
    flat_parvalues = flatview(parvalues)
    idxs = _view_range(axes(flat_parvalues, 1), pa)
    view(flat_parvalues, first(idxs), :)
end

Base.@propagate_inbounds function (pa::ParamDataAccessor{1})(parvalues::AbstractVectorOfSimilarVectors)
    flat_parvalues = flatview(parvalues)
    idxs = _view_range(axes(flat_parvalues, 1), pa)
    fpview = view(flat_parvalues, idxs, :)
    VectorOfSimilarVectors(fpview)
end

Base.@propagate_inbounds function (pa::ParamDataAccessor{N})(parvalues::AbstractVectorOfSimilarVectors) where {N}
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
    ParamShapes{N,AC}

Defines the shapes of a set of named parameters.

Scalar parameters have shape `()`, array parameters have shape `(dim1, dim2, ...)`.

Constructor:

    ParamShapes(name1 = shape1, ...)

e.g.

    parshapes = ParamShapes(a = (2,3), b = (), c = (4,))

Use

    (parshapes::ParamShapes)(data::AbstractVector)::NamedTuple

to get correctly named and shaped views into a flattened parameter vector.
In return,

    Base.Vector{T}(::UndefInitializer, parshapes::ParamShapes)

will create a suitable uninitialized flattened parameter vector. When
dealing with multiple parameter vectors,

    (parshapes::ParamShapes)(
        data::ArrayOfArrays.AbstractVectorOfSimilarVectors
    )::NamedTuple

will transform a vector of flattened parameter vectors into a named tuple of
vectors that contain the (possibly array-shaped) parameter value views as
entries. The resulting tuple is suitable to be wrapped into a
`TypedTables.Table` or similar.

In return,

    ArraysOfArrays.VectorOfSimilarVectors{T}(parshapes::ParamShapes)

will create a suitable vector (of length zero) of flattened parameter vectors.
The result will be a `VectorOfSimilarVectors` wrapped around a 2-dimensional
`ElasticArray`, so all data is stored in a single vector, contiguous in
memory.

Example:

```julia
parshapes = ParamShapes(a = (2,3), b = (), c = (4,))
data = VectorOfSimilarVectors{Int}(parshapes)
resize!(data, 10)
rand!(flatview(data), 0:99)
table = TypedTables.Table(parshapes(data))
fill!(table.b, 42)
all(x -> x == 42, view(flatview(data), 7, :))
```
"""
struct ParamShapes{N,AC}
    _accessors::AC
    _n_pars_flattened::Int

    @inline function ParamShapes(param_shapes::NamedTuple{PN,<:NTuple{N,ParamShapeTuple}}) where {PN,N}
        labels = keys(param_shapes)
        shapes = values(param_shapes)
        lens = map(prod, shapes)
        offsets = _paroffset_cumsum(lens)
        accessors = map(ParamDataAccessor, shapes, offsets)
        lens = map(x -> x.len, accessors)
        n_flattened = sum(lens)
        named_accessors = NamedTuple{labels}(accessors)
        new{PN,typeof(named_accessors)}(named_accessors, n_flattened)
    end
end

export ParamShapes

@inline ParamShapes(;param_sizes...) = ParamShapes(param_sizes.data)


@inline _accessors(x::ParamShapes) = getfield(x, :_accessors)
@inline _n_pars_flattened(x::ParamShapes) = getfield(x, :_n_pars_flattened)


@inline Base.keys(parshapes::ParamShapes) = keys(_accessors(parshapes))

@inline Base.getproperty(parshapes::ParamShapes, p::Symbol) = getfield(_accessors(parshapes), p)

@inline Base.propertynames(parshapes::ParamShapes) = keys(parshapes)


Base.@propagate_inbounds function (parshapes::ParamShapes)(parvalues::AbstractVector)
    accessors = _accessors(parshapes)
    map(pa -> pa(parvalues), accessors) # Not type-stable, investigate!
end

Base.@propagate_inbounds function (parshapes::ParamShapes)(parvalues::AbstractVectorOfSimilarVectors)
    accessors = _accessors(parshapes)
    cols = map(pa -> pa(parvalues), accessors) # Not type-stable, investigate!
    cols
end


StatsBase.dof(parshapes::ParamShapes) = _n_pars_flattened(parshapes)


Base.Vector{T}(::UndefInitializer, parshapes::ParamShapes) where T =
    Vector{T}(undef, StatsBase.dof(parshapes))


ArraysOfArrays.VectorOfSimilarVectors{T}(parshapes::ParamShapes) where T =
    VectorOfSimilarVectors(ElasticArray{T}(undef, StatsBase.dof(parshapes), 0))
