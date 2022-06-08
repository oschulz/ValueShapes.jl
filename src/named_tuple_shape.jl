# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


@inline _varoffset_cumsum_impl(s, x, y, rest...) = (s, _varoffset_cumsum_impl(s+x, y, rest...)...)
@inline _varoffset_cumsum_impl(s,x) = (s,)
@inline _varoffset_cumsum_impl(s) = ()
@inline _varoffset_cumsum(x::Tuple) = _varoffset_cumsum_impl(0, x...)


"""
    NamedTupleShape{names,...} <: AbstractValueShape

Defines the shape of a `NamedTuple` (resp.  set of variables, parameters,
etc.).

Constructors:

    NamedTupleShape(name1 = shape1::AbstractValueShape, ...)
    NamedTupleShape(named_shapes::NamedTuple)

Example:

```julia
shape = NamedTupleShape(
    a = ScalarShape{Real}(),
    b = ArrayShape{Real}(2, 3),
    c = ConstValueShape(42)
)

data = VectorOfSimilarVectors{Float64}(shape)
resize!(data, 10)
rand!(flatview(data))
table = shape.(data)
fill!(table.a, 4.2)
all(x -> x == 4.2, view(flatview(data), 1, :))
```

See also the documentation of [`AbstractValueShape`](@ref).
"""
struct NamedTupleShape{names,AT<:(NTuple{N,ValueAccessor} where N),VT} <: AbstractValueShape
    _accessors::NamedTuple{names,AT}
    _flatdof::Int

    @inline function NamedTupleShape{names,AT,VT}(
        _accessors::NamedTuple{names,AT}, _flatdof::Int
    ) where {names,AT,VT}
        new{names,AT,VT}(_accessors, _flatdof)
    end

    @inline function NamedTupleShape(::Type{VT}, shape::NamedTuple{names,<:NTuple{N,AbstractValueShape}}) where {VT,names,N}
        labels = keys(shape)
        shapes = values(shape)
        shapelengths = map(totalndof, shapes)
        offsets = _varoffset_cumsum(shapelengths)
        accessors = map(ValueAccessor, shapes, offsets)
        # acclengths = map(x -> x.len, accessors)
        # @assert shapelengths == acclengths
        n_flattened = sum(shapelengths)
        named_accessors = NamedTuple{labels}(accessors)
        new{names,typeof(accessors),VT}(named_accessors, n_flattened)
    end
end

export NamedTupleShape

@inline NamedTupleShape(::Type{VT}; named_shapes...) where VT = NamedTupleShape(VT, values(named_shapes))

@inline NamedTupleShape(shape::NamedTuple{names,<:NTuple{N,AbstractValueShape}}) where {names,N} = NamedTupleShape(NamedTuple, shape)
@inline NamedTupleShape(;named_shapes...) = NamedTupleShape(NamedTuple;named_shapes...)


@inline _accessors(x::NamedTupleShape) = getfield(x, :_accessors)
@inline _flatdof(x::NamedTupleShape) = getfield(x, :_flatdof)

function Base.show(io::IO, shape::NamedTupleShape{names,AT,VT}) where {names,AT,VT}
    print(io, Base.typename(typeof(shape)).name, "(")
    if !(VT <: NamedTuple)
        show(io, VT)
        print(io, ", ")
    end
    show(io, (;shape...))
    print(io, ")")
end

Base.:(==)(a::NamedTupleShape, b::NamedTupleShape) = _accessors(a) == _accessors(b)
Base.isequal(a::NamedTupleShape, b::NamedTupleShape) = isequal(_accessors(a), _accessors(b))
#Base.isapprox(a::NamedTupleShape, b::NamedTupleShape; kwargs...) = ...
Base.hash(x::NamedTupleShape, h::UInt) = hash(_accessors(x), hash(:NamedTupleShape, hash(:ValueShapes, h)))


@inline totalndof(shape::NamedTupleShape) = _flatdof(shape)

@inline Base.keys(shape::NamedTupleShape) = keys(_accessors(shape))

@inline Base.values(shape::NamedTupleShape) = values(_accessors(shape))

@inline Base.getindex(d::NamedTupleShape, k::Symbol) = _accessors(d)[k]

@inline function Base.getproperty(shape::NamedTupleShape, p::Symbol)
    # Need to include internal fields of NamedTupleShape to make Zygote happy (ToDo: still true?):
    if p == :_accessors
        getfield(shape, :_accessors)
    elseif p == :_flatdof
        getfield(shape, :_flatdof)
    else
        getproperty(_accessors(shape), p)
    end
end

@inline function Base.propertynames(shape::NamedTupleShape, private::Bool = false)
    names = propertynames(_accessors(shape))
    if private
        (names..., :_flatdof, :_accessors)
    else
        names
    end
end

@inline Base.length(shape::NamedTupleShape) = length(_accessors(shape))

@inline Base.getindex(shape::NamedTupleShape, i::Integer) = getindex(_accessors(shape), i)

@inline Base.map(f, shape::NamedTupleShape) = map(f, _accessors(shape))


function Base.merge(a::NamedTuple, shape::NamedTupleShape{names}) where {names}
    merge(a, NamedTuple{names}(map(x -> valshape(x), values(shape))))
end

Base.merge(a::NamedTupleShape) = a
function Base.merge(a::NamedTupleShape{names,AT,VT}, b::NamedTupleShape, cs::NamedTupleShape...) where {names,AT,VT}
    merge(NamedTupleShape(VT; a..., b...), cs...)
end

function Base.:(<=)(a::NamedTupleShape{names}, b::NamedTupleShape{names}) where {names}
    all(map((a, b) -> a.offset == b.offset && a.shape <= b.shape, values(a), values(b)))
end

@inline Base.:(<=)(a::NamedTupleShape, b::NamedTupleShape) = false


valshape(x::NamedTuple) = NamedTupleShape(NamedTuple, map(valshape, x))


(shape::NamedTupleShape)(::UndefInitializer) = map(x -> valshape(x)(undef), shape)


@inline _multi_promote_type() = Nothing
@inline _multi_promote_type(T::Type) = T
@inline _multi_promote_type(T::Type, U::Type, rest::Type...) = promote_type(T, _multi_promote_type(U, rest...))


@inline default_unshaped_eltype(shape::NamedTupleShape) =
    _multi_promote_type(map(default_unshaped_eltype, values(shape))...)


function unshaped(x::NamedTuple{names}, shape::NamedTupleShape{names}) where names
    # ToDo: Improve performance of return type inference
    T = default_unshaped_eltype(shape)
    U = default_unshaped_eltype(valshape(x))
    R = promote_type(T, U)

    x_unshaped = Vector{R}(undef, totalndof(shape)...)
    sntshape = NamedTupleShape(ShapedAsNT; shape...)
    sntshape(x_unshaped)[] = x
    x_unshaped
end

function unshaped(x::AbstractArray{<:NamedTuple{names},0}, shape::NamedTupleShape{names}) where names
    unshaped(x[], shape)
end


function replace_const_shapes(f::Function, shape::NamedTupleShape{names,AT,VT}) where {names,AT,VT}
    NamedTupleShape(VT, map(s -> replace_const_shapes(f, s), (;shape...)))
end



"""
    ShapedAsNT{names,...}

View of an `AbstractVector{<:Real}` as a mutable named tuple (though not) a
`NamedTuple`, exactly), according to a specified [`NamedTupleShape`](@ref).

Constructors:

    ShapedAsNT(data::AbstractVector{<:Real}, shape::NamedTupleShape)

    shape(data)

The resulting `ShapedAsNT` shares memory with `data`:

```julia
x = (a = 42, b = rand(1:9, 2, 3))
shape = NamedTupleShape(
    ShapedAsNT,
    a = ScalarShape{Real}(),
    b = ArrayShape{Real}(2, 3)
)
data = Vector{Int}(undef, shape)
y = shape(data)
@assert y isa ShapedAsNT
y[] = x
@assert y[] == x
y.a = 22
@assert shape(data) == y
@assert unshaped(y) === data
```

Use `unshaped(x)` to access `data` directly.

See also [`ShapedAsNTArray`](@ref).
"""
struct ShapedAsNT{names,D<:AbstractVector{<:Real},S<:NamedTupleShape{names}}
    __internal_data::D
    __internal_valshape::S

    function ShapedAsNT{names,D,S}(__internal_data::D, __internal_valshape::S) where {names,D<:AbstractVector{<:Real},S<:NamedTupleShape{names}}
        new{names,D,S}(__internal_data, __internal_valshape)
    end

    Base.@propagate_inbounds function ShapedAsNT(data::D, shape::S) where {T<:Real,D<:AbstractVector{T},names,S<:NamedTupleShape{names}}
        @boundscheck _checkcompat(shape, data)
        fixed_shape = _snt_ntshape(shape)
        new{names,D,typeof(fixed_shape)}(data, fixed_shape)
    end
end

export ShapedAsNT

_snt_ntshape(vs::NamedTupleShape{names,AT,<:ShapedAsNT}) where {names,AT} = vs
_snt_ntshape(vs::NamedTupleShape{names,AT}) where {names,AT} = NamedTupleShape{names,AT,ShapedAsNT}(_accessors(vs), _flatdof(vs))


@inline shaped_type(shape::NamedTupleShape{names,AT,<:NamedTuple}, ::Type{T}) where {names,AT,T<:Real} =
    NamedTuple{names,Tuple{map(acc -> shaped_type(acc.shape, T), values(_accessors(shape)))...}}

@inline shaped_type(shape::NamedTupleShape{names,AT,<:ShapedAsNT}, ::Type{T}) where {names,AT,T<:Real} =
    ShapedAsNT{names,Vector{T},typeof(shape)}


Base.@propagate_inbounds (shape::NamedTupleShape{names,AT,<:NamedTuple})(
    data::AbstractVector{<:Real}
) where {names,AT} = ShapedAsNT(data, shape)[]

Base.@propagate_inbounds (shape::NamedTupleShape{names,AT,<:ShapedAsNT})(
    data::AbstractVector{<:Real}
) where {names,AT} = ShapedAsNT(data, shape)


@inline _data(A::ShapedAsNT) = getfield(A, :__internal_data)
@inline _valshape(A::ShapedAsNT) = getfield(A, :__internal_valshape)

@inline valshape(A::ShapedAsNT) = _valshape(A)
@inline unshaped(A::ShapedAsNT) = _data(A)


function unshaped(x::ShapedAsNT{names}, shape::NamedTupleShape{names}) where names
    valshape(x) <= shape || throw(ArgumentError("Shape of value not compatible with given shape"))
    unshaped(x)
end


@inline _shapedasnt_getprop(data::AbstractArray{<:Real}, va::ValueAccessor) = view(data, va)
@inline _shapedasnt_getprop(data::AbstractArray{<:Real}, va::ScalarAccessor) = getindex(data, va)

# ToDo: Move index calculation to separate function with no-op custom pullback to increase performance?
Base.@propagate_inbounds function Base.getproperty(A::ShapedAsNT, p::Symbol)
    # Need to include internal fields of ShapedAsNT to make Zygote happy (ToDo: still true?):
    if p == :__internal_data
        getfield(A, :__internal_data)
    elseif p == :__internal_valshape
        getfield(A, :__internal_valshape)
    else
        data = _data(A)
        shape = _valshape(A)
        va = getproperty(_accessors(shape), p)
        _shapedasnt_getprop(data, va)
    end
end

Base.@propagate_inbounds function Base.setproperty!(A::ShapedAsNT, p::Symbol, x)
    data = _data(A)
    shape = _valshape(A)
    va = getproperty(_accessors(shape), p)
    setindex!(data, x, va)
    A
end

Base.@propagate_inbounds function Base.setproperty!(A::ShapedAsNT, p::Symbol, x::ZeroTangent)
    data = _data(A)
    shape = _valshape(A)
    va = getproperty(_accessors(shape), p)
    idxs = view_idxs(eachindex(data), va)
    fill!(view(data, idxs), zero(eltype(data)))
    A
end

@inline function Base.propertynames(A::ShapedAsNT, private::Bool = false)
    names = Base.propertynames(_valshape(A))
    if private
        (names..., :__internal_data, :__internal_valshape)
    else
        names
    end
end


Base.size(x::ShapedAsNT) = ()
Base.axes(x::ShapedAsNT) = ()
Base.length(x::ShapedAsNT) = 1
Base.isempty(x::ShapedAsNT) = false
Base.ndims(x::ShapedAsNT) = 0
Base.ndims(::Type{<:ShapedAsNT}) = 0
Base.iterate(r::ShapedAsNT) = (r[], nothing)
Base.iterate(r::ShapedAsNT, s) = nothing
Base.IteratorSize(::Type{<:ShapedAsNT}) = HasShape{0}()


Base.@propagate_inbounds function Base.getindex(x::ShapedAsNT{names}) where names
    if @generated
        Expr(:tuple, map(p -> :($p = x.$p), names)...)
    else
        # Shouldn't be used, ideally
        @assert false
        accessors = _accessors(_valshape(x))
        data = _data(x)
        map(va -> getindex(data, va), accessors)
    end
end

@inline Base.getindex(d::ShapedAsNT, k::Symbol) = getproperty(d, k)


Base.@propagate_inbounds Base.view(A::ShapedAsNT) = A


Base.@propagate_inbounds function Base.setindex!(A::ShapedAsNT{names}, x::NamedTuple{names}) where {names}
    if @generated
        Expr(:block, map(p -> :(A.$p = x.$p), names)...)
    else
        # Shouldn't be used, ideally
        @assert false
        map(n -> setproperty!(A, n, getproperty(x, n)), names)
    end

    A
end

Base.@propagate_inbounds Base.setindex!(A::ShapedAsNT{T}, x) where {T} = setindex!(A, convert(T, x))

Base.@propagate_inbounds function Base.setindex!(A::ShapedAsNT, x, i::Integer)
    @boundscheck Base.checkbounds(A, i)
    setindex!(A, x)
end


Base.NamedTuple(A::ShapedAsNT) = A[]
Base.NamedTuple{names}(A::ShapedAsNT{names}) where {names} = A[]

Base.convert(::Type{NamedTuple}, A::ShapedAsNT) = A[]
Base.convert(::Type{NamedTuple{names}}, A::ShapedAsNT{names}) where {names} = A[]

function Base.convert(::Type{ShapedAsNT{names,D_a,S}}, A::ShapedAsNT{names,D_b,S}) where {names,D_a,D_b,S}
    ShapedAsNT{names,D_a,S}(convert(D_a,_data(A)), valshape(A))
end

realnumtype(::Type{<:ShapedAsNT{<:Any,<:AbstractArray{T}}}) where {T<:Real} = T

stripscalar(A::ShapedAsNT) = A[]

Base.show(io::IO, ::MIME"text/plain", A::ShapedAsNT) = show(io, A)

function Base.show(io::IO, A::ShapedAsNT)
    print(io, Base.typename(typeof(A)).name, "(")
    show(io, A[])
    print(io, ")")
end

Base.:(==)(A::ShapedAsNT, B::ShapedAsNT) = _data(A) == _data(B) && _valshape(A) == _valshape(B)
Base.isequal(A::ShapedAsNT, B::ShapedAsNT) = isequal(_data(A), _data(B)) && _valshape(A) == _valshape(B)
Base.isapprox(A::ShapedAsNT, B::ShapedAsNT; kwargs...) = isapprox(_data(A), _data(B); kwargs...) && _valshape(A) == _valshape(B)

Base.copy(A::ShapedAsNT) = ShapedAsNT(copy(_data(A)),_valshape(A))


# Required for accumulation during automatic differentiation:
function Base.:(+)(A::ShapedAsNT{names}, B::ShapedAsNT{names}) where names
    @argcheck _valshape(A) == _valshape(B)
    ShapedAsNT(_data(A) + _data(B), _valshape(A))
end

# Required for accumulation during automatic differentiation:
function ChainRulesCore.add!!(A::ShapedAsNT{names}, B::ShapedAsNT{names}) where names
    @argcheck _valshape(A) == _valshape(B)
    ChainRulesCore.add!!(_data(A), _data(B))
    return A
end


function ChainRulesCore.Tangent(x::T, unshaped_dx::AbstractVector{<:Real}) where {T<:ShapedAsNT}
    vs = valshape(x)
    gs = gradient_shape(vs)
    contents = (__internal_data = unshaped_dx, __internal_valshape = gs)
    Tangent{T,typeof(contents)}(contents)
end


struct GradShapedAsNTProjector{VS<:NamedTupleShape} <: Function
    gradshape::VS
end

ChainRulesCore.ProjectTo(x::ShapedAsNT) = GradShapedAsNTProjector(gradient_shape(valshape(x)))


_check_ntgs_tangent_compat(a::NamedTupleShape, ::NoTangent) = nothing
function _check_ntgs_tangent_compat(a::NamedTupleShape{names}, b::NamedTupleShape{names}) where names
    a >= b || error("Incompatible tangent NamedTupleShape")
end

_snt_from_tangent(data::AbstractVector{<:Real}, gs::NamedTupleShape) = ShapedAsNT(data, gs)
_snt_from_tangent(data::_ZeroLike, gs::NamedTupleShape) = _az_tangent(data)

function (project::GradShapedAsNTProjector{<:NamedTupleShape{names}})(data::NamedTuple{(:__internal_data, :__internal_valshape)}) where names
    gs = project.gradshape
    _check_ntgs_tangent_compat(gs, data.__internal_valshape)
    _snt_from_tangent(data.__internal_data, project.gradshape)
end

function (project::GradShapedAsNTProjector{<:NamedTupleShape{names}})(tangent::Tangent{<:ShapedAsNT{names}}) where names
    project(_backing(tangent))
end

function (project::GradShapedAsNTProjector{<:NamedTupleShape{names}})(tangent::ShapedAsNT{names}) where names
    tangent
end

(project::GradShapedAsNTProjector{<:NamedTupleShape})(tangent::_ZeroLike) = _az_tangent(tangent)


_getindex_tangent(x::ShapedAsNT, dy::_ZeroLike) = _az_tangent(dy)

function _getindex_tangent(x::ShapedAsNT, dy::NamedTuple)
    tangent = Tangent(x, _tangent_array(unshaped(x)))
    dx_unshaped, gs = _backing(tangent)
    ShapedAsNT(dx_unshaped, gs)[] = _notangent_to_zerotangent(dy)
    tangent
end

function ChainRulesCore.rrule(::typeof(getindex), x::ShapedAsNT)
    shapedasnt_getindex_pullback(ΔΩ) = (NoTangent(), ProjectTo(x)(_getindex_tangent(x, _unpack_tangent(ΔΩ))))
    return x[], shapedasnt_getindex_pullback
end


_unshaped_tangent(x::ShapedAsNT, dy::AbstractArray{<:Real}) = Tangent(x, dy)
_unshaped_tangent(x::ShapedAsNT, dy::_ZeroLike) = _az_tangent(dy)

function ChainRulesCore.rrule(::typeof(unshaped), x::ShapedAsNT)
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), ProjectTo(x)(_unshaped_tangent(x, _unpack_tangent(ΔΩ))))
    return unshaped(x), unshaped_nt_pullback
end

function ChainRulesCore.rrule(::typeof(unshaped), x::ShapedAsNT, vs::NamedTupleShape)
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), ProjectTo(x)(_unshaped_tangent(x, _unpack_tangent(ΔΩ))), NoTangent())
    return unshaped(x, vs), unshaped_nt_pullback
end

function _unshaped_tangent(x::NamedTuple, vs::NamedTupleShape, dy::AbstractArray{<:Real})
    gs = gradient_shape(vs)
    # gs(dy) can be a NamedTuple or a ShapedAsNT, depending on vs:
    dx = convert(NamedTuple, gs(dy))
    Tangent{typeof(x),typeof(dx)}(dx)
end

_unshaped_tangent(x::NamedTuple, vs::NamedTupleShape, dy::_ZeroLike) = _az_tangent(dy)

function ChainRulesCore.rrule(::typeof(unshaped), x::NamedTuple, vs::NamedTupleShape)
    unshaped_nt_pullback(ΔΩ) = (NoTangent(), _unshaped_tangent(x, vs, _unpack_tangent(ΔΩ)), NoTangent())
    return unshaped(x, vs), unshaped_nt_pullback
end


_shapedasnt_tangent(dy::_ZeroLike, vs::NamedTupleShape{names}) where names = _az_tangent(dy)

_shapedasnt_tangent(dy::ShapedAsNT{names}, vs::NamedTupleShape{names}) where names = unshaped(dy)

function _shapedasnt_tangent(
    dy::Tangent{<:NamedTuple{names},<:NamedTuple{names}},
    vs::NamedTupleShape{names}
) where names
    unshaped(_backing(dy), gradient_shape(vs))
end

function _shapedasnt_tangent(
    dy::Tangent{<:Any,<:NamedTuple{(:__internal_data, :__internal_valshape)}},
    vs::NamedTupleShape{names}
) where names
    _backing(dy).__internal_data
end

function ChainRulesCore.rrule(::Type{ShapedAsNT}, A::AbstractVector{<:Real}, vs::NamedTupleShape{names}) where names
    shapedasnt_pullback(ΔΩ) = (NoTangent(), _shapedasnt_tangent(unthunk(ΔΩ), vs), NoTangent())
    return ShapedAsNT(A, vs), shapedasnt_pullback
end



"""
    ShapedAsNTArray{T<:NamedTuple,...} <: AbstractArray{T,N}

View of an `AbstractArray{<:AbstractVector{<:Real},N}` as an array of
`NamedTuple`s, according to a specified [`NamedTupleShape`](@ref).

`ShapedAsNTArray` implements the `Tables` API.

Constructors:

    ShapedAsNTArray(
        data::AbstractArray{<:AbstractVector{<:Real},
        shape::NamedTupleShape
    )

    shape.(data)

The resulting `ShapedAsNTArray` shares memory with `data`:

```julia
using ValueShapes, ArraysOfArrays, Tables, TypedTables

X = [
    (a = 42, b = rand(1:9, 2, 3))
    (a = 11, b = rand(1:9, 2, 3))
]

shape = valshape(X[1])
data = nestedview(Array{Int}(undef, totalndof(shape), 2))
Y = shape.(data)
@assert Y isa ShapedAsNTArray
Y[:] = X
@assert Y[1] == X[1] == shape(data[1])
@assert Y.a == [42, 11]
Tables.columns(Y)
@assert unshaped.(Y) === data
@assert Table(Y) isa TypedTables.Table
```

Use `unshaped.(Y)` to access `data` directly.

`Tables.columns(Y)` will return a `NamedTuple` of columns. They will contain
a copy the data, using a memory layout as contiguous as possible for each
column.
"""
struct ShapedAsNTArray{T,N,D<:AbstractArray{<:AbstractVector{<:Real},N},S<:NamedTupleShape} <: AbstractArray{T,N}
    __internal_data::D
    __internal_elshape::S
end

export ShapedAsNTArray


function ShapedAsNTArray(data::D, shape::S) where {N,T<:Real,D<:AbstractArray{<:AbstractVector{T},N},S<:NamedTupleShape}
    NT_T = shaped_type(shape, T)
    ShapedAsNTArray{NT_T,N,D,S}(data, shape)
end


Base.Broadcast.broadcasted(vs::NamedTupleShape, A::AbstractArray{<:AbstractVector{<:Real}}) =
    ShapedAsNTArray(A, vs)


@inline _data(A::ShapedAsNTArray) = getfield(A, :__internal_data)
@inline _elshape(A::ShapedAsNTArray) = getfield(A, :__internal_elshape)

@inline elshape(A::ShapedAsNTArray) = _elshape(A)

realnumtype(::Type{<:ShapedAsNTArray{<:Any,N,<:AbstractArray{<:AbstractArray{T}}}}) where {T<:Real,N} = T


Base.Broadcast.broadcasted(::typeof(identity), A::ShapedAsNTArray) = A

Base.Broadcast.broadcasted(::typeof(unshaped), A::ShapedAsNTArray) = _data(A)

function Base.Broadcast.broadcasted(::typeof(unshaped), A::ShapedAsNTArray, vsref::Ref{<:AbstractValueShape})
    @_adignore elshape(A) <= vsref[] || throw(ArgumentError("Shape of value not compatible with given shape"))
    _data(A)
end


@inline function Base.getproperty(A::ShapedAsNTArray, p::Symbol)
    # Need to include internal fields of ShapedAsNTArray to make Zygote happy (ToDo: still true?):
    if p == :__internal_data
        getfield(A, :__internal_data)
    elseif p == :__internal_elshape
        getfield(A, :__internal_elshape)
    else
        data = _data(A)
        shape = _elshape(A)
        va = getproperty(_accessors(shape), p)
        view.(data, Ref(va))
    end
end

@inline function Base.propertynames(A::ShapedAsNTArray, private::Bool = false)
    names = Base.propertynames(_elshape(A))
    if private
        (names..., :__internal_data, :__internal_elshape)
    else
        names
    end
end

Base.:(==)(A::ShapedAsNTArray, B::ShapedAsNTArray) = _data(A) == _data(B) && _elshape(A) == _elshape(B)
Base.isequal(A::ShapedAsNTArray, B::ShapedAsNTArray) = isequal(_data(A), _data(B)) && _elshape(A) == _elshape(B)
Base.isapprox(A::ShapedAsNTArray, B::ShapedAsNTArray; kwargs...) = isapprox(_data(A), _data(B); kwargs...) && _elshape(A) == _elshape(B)


@inline Base.size(A::ShapedAsNTArray) = size(_data(A))
@inline Base.axes(A::ShapedAsNTArray) = axes(_data(A))
@inline Base.IndexStyle(::Type{<:ShapedAsNTArray{T,N,D}}) where {T,N,D} = IndexStyle(D)


ShapedAsNT(A::ShapedAsNTArray{T,0}) where T = _elshape(A)(first(_data(A)))
ShapedAsNT{names}(A::ShapedAsNTArray{<:NamedTuple{names},0}) where {names,T} = _elshape(A)(first(_data(A)))

Base.convert(::Type{ShapedAsNT}, A::ShapedAsNTArray{T,0}) where T = ShapedAsNT(A)
Base.convert(::Type{ShapedAsNT{names}}, A::ShapedAsNTArray{<:NamedTuple{names},0}) where {names,T} = ShapedAsNT{names}(A)


Base.@propagate_inbounds _apply_ntshape_copy(data::AbstractVector{<:Real}, shape::NamedTupleShape) = shape(data)


Base.@propagate_inbounds _apply_ntshape_copy(data::AbstractArray{<:AbstractVector{<:Real}}, shape::NamedTupleShape) =
    ShapedAsNTArray(data, shape)

Base.getindex(A::ShapedAsNTArray, idxs...) = _apply_ntshape_copy(getindex(_data(A), idxs...), _elshape(A))

Base.view(A::ShapedAsNTArray, idxs...) = ShapedAsNTArray(view(_data(A), idxs...), _elshape(A))


function Base.setindex!(A::ShapedAsNTArray, x, idxs::Integer...)
    A_idxs = ShapedAsNT(getindex(_data(A), idxs...), _elshape(A))
    setindex!(A_idxs, x)
end


function Base.similar(A::ShapedAsNTArray{T}, ::Type{T}, dims::Dims) where T
    data = _data(A)
    U = eltype(data)
    newdata = similar(data, U, dims)
    # In case newdata is not something like an ArrayOfSimilarVectors:
    if !isempty(newdata) && !isdefined(newdata, firstindex(newdata))
        for i in eachindex(newdata)
            newdata[i] = similar(data[firstindex(data)])
        end
    end
    ShapedAsNTArray(newdata, _elshape(A))
end


Base.empty(A::ShapedAsNTArray{T,N,D,S}) where {T,N,D,S} =
    ShapedAsNTArray{T,N,D,S}(empty(_data(A)), _elshape(A))

# For some reason `TypedTables.columnnames` is a different function than Tables.columnnames(A),
# `TypedTables.columnnames` doesn't support arrays of ShapedAsNT, so define:
TypedTables.columnnames(A::ShapedAsNTArray) = Tables.columnnames(A)

Base.show(io::IO, ::MIME"text/plain", A::ShapedAsNTArray) = TypedTables.showtable(io, A)

function Base.show(io::IO, ::MIME"text/plain", A::ShapedAsNTArray{T,0}) where T
    println(io, "0-dimensional ShapedAsNTArray:")
    show(io, A[])
end


Base.copy(A::ShapedAsNTArray) = ShapedAsNTArray(copy(_data(A)), _elshape(A))


Base.pop!(A::ShapedAsNTArray) = _elshape(A)(pop!(_data(A)))

# Base.push!(A::ShapedAsNTArray, x::Any)  # ToDo


Base.popfirst!(A::ShapedAsNTArray) = _elshape(A)(popfirst!(_data(A)))

# Base.pushfirst!(A::ShapedAsNTArray, x::Any)  # ToDo


function Base.append!(A::ShapedAsNTArray, B::ShapedAsNTArray)
    _elshape(A) == _elshape(B) || throw(ArgumentError("Can't append ShapedAsNTArray instances with different element shapes"))
    append!(_data(A), _data(B))
    A
end

# Base.append!(A::ShapedAsNTArray, B::AbstractArray)  # ToDo


function Base.prepend!(A::ShapedAsNTArray, B::ShapedAsNTArray)
    _elshape(A) == _elshape(B) || throw(ArgumentError("Can't prepend ShapedAsNTArray instances with different element shapes"))
    prepend!(_data(A), _data(B))
    A
end

# Base.prepend!(A::ShapedAsNTArray, B::AbstractArray)  # ToDo


function Base.deleteat!(A::ShapedAsNTArray, i)
    deleteat!(_data(A), i)
    A
end

# Base.insert!(A::ShapedAsNTArray, i::Integer, x::Any)  # ToDo


Base.splice!(A::ShapedAsNTArray, i) = _elshape(A)(splice!(_data(A), i))

# Base.splice!(A::ShapedAsNTArray, i, replacement)  # ToDo


function Base.vcat(A::ShapedAsNTArray, B::ShapedAsNTArray)
    _elshape(A) == _elshape(B) || throw(ArgumentError("Can't vcat ShapedAsNTArray instances with different element shapes"))
    ShapedAsNTArray(vcat(_data(A), _data(B)), _elshape(A))
end

# Base.vcat(A::ShapedAsNTArray, B::AbstractArray)  # ToDo


# Base.hcat(A::ShapedAsNTArray, B) # ToDo


Base.vec(A::ShapedAsNTArray{T,1}) where T = A
Base.vec(A::ShapedAsNTArray) = ShapedAsNTArray(vec(_data(A)), _elshape(A))


Tables.istable(::Type{<:ShapedAsNTArray}) = true
Tables.rowaccess(::Type{<:ShapedAsNTArray}) = true
Tables.columnaccess(::Type{<:ShapedAsNTArray}) = true
Tables.schema(A::ShapedAsNTArray{T}) where {T} = Tables.Schema(T)

function Tables.columns(A::ShapedAsNTArray)
    data = _data(A)
    accessors = _accessors(_elshape(A))
    # Copy columns to make each column as contiguous in memory as possible:
    map(va -> getindex.(data, Ref(va)), accessors)
end

@inline Tables.rows(A::ShapedAsNTArray) = A



const _AnySNTArray{names} = ShapedAsNTArray{<:Union{NamedTuple{names},ShapedAsNT{names}}}


# For accumulation during automatic differentiation:
function Base.:(+)(A::_AnySNTArray{names}, B::_AnySNTArray{names}) where names
    @argcheck elshape(A) == elshape(B)
    ShapedAsNTArray(_data(A) + _data(B), elshape(A))
end

# For accumulation during automatic differentiation:
function ChainRulesCore.add!!(A::_AnySNTArray{names}, B::_AnySNTArray{names}) where names
    @argcheck elshape(A) == elshape(B)
    ChainRulesCore.add!!(_data(A), _data(B))
    return A
end


function ChainRulesCore.Tangent(X::T, unshaped_dX::AbstractArray{<:AbstractVector{<:Real}}) where {T<:ShapedAsNTArray}
    vs = elshape(X)
    gs = gradient_shape(vs)
    contents = (__internal_data = unshaped_dX, __internal_elshape = gs)
    Tangent{T,typeof(contents)}(contents)
end


struct GradShapedAsNTArrayProjector{VS<:NamedTupleShape} <: Function
    gradshape::VS
end

ChainRulesCore.ProjectTo(X::ShapedAsNTArray) = GradShapedAsNTArrayProjector(gradient_shape(elshape(X)))

function (project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(data::NamedTuple{(:__internal_data, :__internal_elshape)}) where names
    _check_ntgs_tangent_compat(project.gradshape, data.__internal_elshape)
    _keep_zerolike(ShapedAsNTArray, data.__internal_data, project.gradshape)
end

(project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(tangent::Tangent{<:_AnySNTArray}) where names = project(_backing(tangent))
(project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(tangent::_AnySNTArray) where names = tangent
(project::GradShapedAsNTArrayProjector{<:NamedTupleShape})(tangent::_ZeroLike) = _az_tangent(tangent)

function (project::GradShapedAsNTArrayProjector{<:NamedTupleShape{names}})(
    tangent::AbstractArray{<:Union{Tangent{<:Any,<:NamedTuple{names}},ShapedAsNT{names}}}
) where names
    data =_shapedasntarray_tangent(tangent, project.gradshape)
    ShapedAsNTArray(data, project.gradshape)
end


_tablecols_tangent(X::ShapedAsNTArray, dY::_ZeroLike) = _az_tangent(dY)

function _write_snta_col!(data::AbstractArray{<:AbstractVector{<:Real}}, va::ValueAccessor, A::AbstractArray)
    B = view.(data, Ref(va))
    B .= A
end
function _write_snta_col!(data::ArrayOfSimilarVectors{<:Real}, va::ValueAccessor, A::_ZeroLike)
    flat_data = flatview(data)
    idxs = view_idxs(axes(flat_data, 1), va)
    fill!(view(flat_data, idxs, :), zero(eltype(flat_data)))
end
_write_snta_col!(data::AbstractArray{<:AbstractVector{<:Real}}, va::ConstAccessor, A) = nothing

function _tablecols_tangent(X::_AnySNTArray, dY::NamedTuple{names}) where names
    tangent = Tangent(X, _tangent_array(_data(X)))
    dx_unshaped, gs = _backing(tangent)
    global g_state = (;X, dY)
    # ToDo: Re-check safety of this after nested-NT arrays are implemented:
    map((va_i, dY_i) -> _write_snta_col!(dx_unshaped, va_i, dY_i), _accessors(gs), _notangent_to_zerotangent(dY))
    tangent
end

g_state = nothing
function ChainRulesCore.rrule(::typeof(Tables.columns), X::ShapedAsNTArray)
    tablecols_pullback(ΔΩ) = begin
        (NoTangent(), ProjectTo(X)(_tablecols_tangent(X, _unpack_tangent(ΔΩ))))
    end
    return Tables.columns(X), tablecols_pullback
end


_data_tangent(X::ShapedAsNTArray, dY::AbstractArray{<:AbstractVector{<:Real}}) = Tangent(X, dY)
_data_tangent(X::ShapedAsNTArray, dY::_ZeroLike) = _az_tangent(dY)

function ChainRulesCore.rrule(::typeof(_data), X::ShapedAsNTArray)
    _data_pullback(ΔΩ) = (NoTangent(), ProjectTo(X)(_data_tangent(X, _unpack_tangent(ΔΩ))))
    return _data(X), _data_pullback
end


_shapedasntarray_tangent(dY::_ZeroLike, vs::NamedTupleShape{names}) where names = _az_tangent(dY)

_shapedasntarray_tangent(dY::_AnySNTArray, vs::NamedTupleShape{names}) where names = unshaped.(dY)

function _shapedasntarray_tangent(
    dY::AbstractArray{<:Tangent{<:Any,<:NamedTuple{names}}},
    vs::NamedTupleShape{names}
) where names
    ArrayOfSimilarArrays(unshaped.(ValueShapes._backing.(dY), Ref(gradient_shape(vs))))
end

function _shapedasntarray_tangent(
    dY::AbstractArray{<:ShapedAsNT{names}},
    vs::NamedTupleShape{names}
) where names
    ArrayOfSimilarArrays(unshaped.(dY))
end

function _shapedasntarray_tangent(
    dY::Tangent{<:Any,<:NamedTuple{(:__internal_data, :__internal_elshape)}},
    vs::NamedTupleShape{names}
) where names
    _backing(dY).__internal_data
end

function ChainRulesCore.rrule(::Type{ShapedAsNTArray}, A::AbstractArray{<:AbstractVector{<:Real}}, vs::NamedTupleShape{names}) where names
    shapedasntarray_pullback(ΔΩ) = begin
        global g_state = (;A, ΔΩ, vs)
        (NoTangent(), _shapedasntarray_tangent(unthunk(ΔΩ), vs), NoTangent())
    end
    return ShapedAsNTArray(A, vs), shapedasntarray_pullback
end
