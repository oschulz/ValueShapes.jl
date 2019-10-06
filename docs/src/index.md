# ValueShapes.jl

ValueShapes provides Julia types to describe the shape of values and
to describe sets of named variables.

This package aims to provide a bridge between user code operating on named
structured variables and (e.g. fitting/optimization) algorithms operating on
flat vectors of anonymous real values. ValueShapes provides a zero-copy
duality of view between both representations.

ValueShapes defines the shape of a value as the combination of it's data
type (resp. element type, in the case of arrays) and the size of the value
(relevant if the value is an array), e.g.

```julia
using ValueShapes

ScalarShape{Real}()
ArrayShape{Real}(2, 3)
ConstValueShape([1 2; 3 4])
```

Array shapes can be used to construct a compatible array:

```julia
Array(undef, ArrayShape{Real}(2, 3))
```

Building on this, ValueShapes provides a way to describe a set of
named variables and their shapes:

```julia
varshapes = VarShapes(
    a = ScalarShape{Real}(),
    b = ArrayShape{Real}(2, 3),
    c = ConstValueShape([1 2; 3 4])
)
```

`a`, `b`, `c` could, e.g., be the parameters of a fit with `c` being held at
a fixed value. This set of variables has

```julia
totalndof(varshapes) == 7
```

total degrees of freedom (the constant `c` does not contribute). Instead of
storing a concrete instance of values of these variables in a nested
structure, the underlying values can also be stored in a real-valued flat
vector of length 7:

```julia
using Random

data = Vector{Float64}(undef, varshapes)
size(data) == (7,)
rand!(data)
```

from which

```julia
tupleview = varshapes(data)
```

will construct a named tuple of the variable values, implemented as views
into the vector `data`:

```julia
typeof(tupleview) <: NamedTuple{(:a, :b, :c)}
```

Note: The package [EponymTuples](https://github.com/tpapp/EponymTuples.jl)
may come in handy to define functions that take such tuples as
parameters and deconstruct them, so that the variable names can be used
directly inside the function body.

ValueShapes can also handle multiple values for sets of variables and
is designed to compose well with
[ArraysOfArrays](https://github.com/oschulz/ArraysOfArrays.jl) and
[TypedTables](https://github.com/FugroRoames/TypedTables.jl)
(and similar table packages):

```julia
using ArraysOfArrays, Tables, TypedTables

multidata = VectorOfSimilarVectors{Int}(varshapes)
resize!(multidata, 10)
rand!(flatview(multidata), 0:99)

table = varshapes(multidata)
keys(Tables.columns(table)) == (:a, :b, :c)
```
