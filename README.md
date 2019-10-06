# ValueShapes.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://oschulz.github.io/ValueShapes.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://oschulz.github.io/ValueShapes.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Travis Build Status](https://travis-ci.com/oschulz/ValueShapes.jl.svg?branch=master)](https://travis-ci.com/oschulz/ValueShapes.jl)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/github/oschulz/ValueShapes.jl?branch=master&svg=true)](https://ci.appveyor.com/project/oschulz/ValueShapes-jl)
[![Codecov](https://codecov.io/gh/oschulz/ValueShapes.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/oschulz/ValueShapes.jl)


## Documentation

ValueShapes provides Julia types to describe the shape of values and
of sets of named variables. The aim is to provide a bridge between user code
operating on named structured variables and (e.g. fitting/optimization)
algorithms operating on flat vectors of anonymous real values.

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
views instead of data copies).
