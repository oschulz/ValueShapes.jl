# ValueShapes.jl

ValueShapes provides Julia types to describe the shape of values, like
scalars, arrays and structures.

Shapes provide a generic way to construct uninitialized values (e.g.
multidimensional arrays) without using templates.

Shapes also act as a bridge between structured and flat data representations:
Mathematical and statistical algorithms (e.g. optimizers, fitters, solvers,
etc.) often represent variables/parameters as flat vectors of nameless real
values. But user code will usually be more concise and readable if
variables/parameters can have names (e.g. via `NamedTuple`s) and non-scalar
shapes. ValueShapes provides a duality of view between the two different data
representations.

ValueShapes defines the shape of a value as the combination of it's data
type (resp. element type, in the case of arrays) and the size of the value
(relevant if the value is an array), e.g.

```julia
using ValueShapes

ScalarShape{Real}()
ArrayShape{Real}(2, 3)
ConstValueShape([1 2; 3 4])
```


Array shapes can be used to construct a compatible real-valued data vector:

```julia
Vector{Float64}(undef, ArrayShape{Real}(2, 3)) isa Vector{Float64}
```

ValueShapes also provides a way to define the shape of a `NamedTuple`.
This can be used to specify the names and shapes of a set of variables or
parameters. Consider a fitting problem with the following parameters: A
scalar `a`, a 2x3 array `b` and an array `c` pinned to a fixed value. This
set parameters can be specified as


```julia
parshapes = NamedTupleShape(
    a = ScalarShape{Real}(),
    b = ArrayShape{Real}(2, 3),
    c = ConstValueShape([1 2; 3 4])
)
```

This set of parameters has

```julia
totalndof(parshapes) == 7
```

total degrees of freedom (the constant `c` does not contribute). The flat data
representation for this `NamedTupleShape` is a vector of length 7:

```julia
using Random

data = Vector{Float64}(undef, parshapes)
size(data) == (7,)
rand!(data)
```

which can again be viewed as a `NamedTuple` described by `shape` via

```julia
data_as_ntuple = parshapes(data)

data_as_ntuple isa ShapedAsNT{(:a, :b, :c)}
```

(See [`ShapedAsNT`](@ref).)

Note: The macro `@unpack` provided by the package
[UnPack](https://github.com/mauro3/UnPack.jl) is very hand to
to unpack `NamedTuple`s selectively.

ValueShapes can also handle multiple values for sets of variables and
is designed to compose well with
[ArraysOfArrays.jl](https://github.com/oschulz/ArraysOfArrays.jl) and
[Tables.jl](https://github.com/JuliaData/Tables.jl)
(and similar table packages). Broadcasting a shape over a vector of
real-valued vectors will create a view that implements the Tables.jl API:

```julia
using ArraysOfArrays, Tables, TypedTables

multidata = VectorOfSimilarVectors{Int}(parshapes)
resize!(multidata, 10)
rand!(flatview(multidata), 0:99)

A = parshapes.(multidata)
keys(Tables.columns(A)) == (:a, :b, :c)
```

ValueShapes supports this via specialized broadcasting. `A` is now a
table-like view into the data (see [`ShapedAsNTArray`](@ref)), and shares
memory with `multidata`. To create an independent `NamedTuple` of columns,
with a contiguous memory layout for each column, use `Tables.columns(A)`:

```julia
tcols = Tables.columns(A)
flatview(tcols.b) isa Array{Int}
```

Constructing a `TypedTables.Table`, `DataFrames.DataFrame` or similar from `A`
will have the same effect:

```julia
using TypedTables
flatview(Table(A).b) isa Array{Int}

using DataFrames
flatview(DataFrame(A).b) isa Array{Int}
```
