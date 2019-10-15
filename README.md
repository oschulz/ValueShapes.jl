# ValueShapes.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://oschulz.github.io/ValueShapes.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://oschulz.github.io/ValueShapes.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Travis Build Status](https://travis-ci.com/oschulz/ValueShapes.jl.svg?branch=master)](https://travis-ci.com/oschulz/ValueShapes.jl)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/github/oschulz/ValueShapes.jl?branch=master&svg=true)](https://ci.appveyor.com/project/oschulz/ValueShapes-jl)
[![Codecov](https://codecov.io/gh/oschulz/ValueShapes.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/oschulz/ValueShapes.jl)


## Documentation

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

See the documentation for details:

* [Documentation for stable version](https://oschulz.github.io/ValueShapes.jl/stable)
* [Documentation for development version](https://oschulz.github.io/ValueShapes.jl/dev)

ValueShapes is designed to compose well with
[ElasticArrays](https://github.com/JuliaArrays/ElasticArrays.jl),
[ArraysOfArrays](https://github.com/oschulz/ArraysOfArrays.jl) and
[TypedTables](https://github.com/FugroRoames/TypedTables.jl) (and similar
table packages). ValueShapes package has some overlap in functionality
with [TransformVariables](https://github.com/tpapp/TransformVariables.jl), but
provides a duality of view instead of transformations (and therefore uses data
views instead of data copies, where possible).
