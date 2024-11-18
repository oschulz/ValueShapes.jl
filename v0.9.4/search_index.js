var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"DocTestSetup  = quote\n    using ValueShapes\nend","category":"page"},{"location":"api/#Modules","page":"API","title":"Modules","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:module]","category":"page"},{"location":"api/#Types-and-constants","page":"API","title":"Types and constants","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:type, :constant]","category":"page"},{"location":"api/#Functions-and-macros","page":"API","title":"Functions and macros","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:macro, :function]","category":"page"},{"location":"api/#Documentation","page":"API","title":"Documentation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [ValueShapes]\nOrder = [:module, :type, :constant, :macro, :function]","category":"page"},{"location":"api/#ValueShapes.ValueShapes","page":"API","title":"ValueShapes.ValueShapes","text":"ValueShapes\n\nProvides a Julia API to describe the shape of values, like scalars, arrays and structures.\n\n\n\n\n\n","category":"module"},{"location":"api/#ValueShapes.AbstractScalarShape","page":"API","title":"ValueShapes.AbstractScalarShape","text":"AbstractScalarShape{T} <: AbstractValueShape\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.AbstractValueShape","page":"API","title":"ValueShapes.AbstractValueShape","text":"abstract type AbstractValueShape\n\nAn AbstractValueShape combines type and size information.\n\nSubtypes are defined for shapes of scalars (see ScalarShape), arrays (see ArrayShape), constant values (see ConstValueShape) and NamedTuples (see NamedTupleShape).\n\nSubtypes of AbstractValueShape must support eltype, size and totalndof.\n\nValue shapes can be used as constructors to generate values of the given shape with undefined content. If the element type of the shape is an abstract or union type, a suitable concrete type will be chosen automatically, if possible (see ValueShapes.default_datatype):\n\nshape = ArrayShape{Real}(2,3)\nA = shape(undef)\ntypeof(A) == Array{Float64,2}\nsize(A) == (2, 3)\nvalshape(A) == ArrayShape{Float64}(2,3)\n\nUse\n\n(shape::AbstractValueShape)(data::AbstractVector{<:Real})::eltype(shape)\n\nto view a flat vector of anonymous real values as a value of the given shape:\n\ndata = [1, 2, 3, 4, 5, 6]\nshape(data) == [1 3 5; 2 4 6]\n\nIn return,\n\nBase.Vector{<:Real}(undef, shape::AbstractValueShape)\n\nwill create a suitable uninitialized vector of the right length to hold such flat data for the given shape. If no type T is given, a suitable data type will be chosen automatically.\n\nWhen dealing with multiple vectors of flattened data, use\n\nshape.(data::ArraysOfArrays.AbstractVectorOfSimilarVectors)\n\nValueShapes supports this via specialized broadcasting.\n\nIn return,\n\nArraysOfArrays.VectorOfSimilarVectors{<:Real}(shape::AbstractValueShape)\n\nwill create a suitable vector (of length zero) of vectors that can hold flattened data for the given shape. The result will be a VectorOfSimilarVectors wrapped around a 2-dimensional ElasticArray. This way, all data is stored in a single contiguous chunk of memory.\n\nAbstractValueShapes can be compared with <= and >=, with semantics that are similar to compare type with <: and >::\n\na::AbstractValueShape <= b::AbstractValueShape == true\n\nimplies that values of shape a are can be used in contexts that expect values of shape b. E.g.:\n\n(ArrayShape{Float64}(4,5) <= ArrayShape{Real}(4,5)) == true\n(ArrayShape{Float64}(4,5) <= ArrayShape{Integer}(4,5)) == false\n(ArrayShape{Float64}(2,2) <= ArrayShape{Float64}(3,3)) == false\n(ScalarShape{Real}() >= ScalarShape{Int}()) == true\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ArrayShape","page":"API","title":"ValueShapes.ArrayShape","text":"ArrayShape{T,N} <: AbstractValueShape\n\nDescribes the shape of N-dimensional arrays of type T and a given size.\n\nConstructor:\n\nArrayShape{T}(dims::NTuple{N,Integer}) where {T,N}\nArrayShape{T}(dims::Integer...) where {T}\n\ne.g.\n\nshape = ArrayShape{Real}(2, 3)\n\nSee also the documentation of AbstractValueShape.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ConstValueDist","page":"API","title":"ValueShapes.ConstValueDist","text":"ConstValueDist <: Distributions.Distribution\n\nRepresents a delta distribution for a constant value of arbritrary type.\n\nCalling varshape on a ConstValueDist will yield a ConstValueShape.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ConstValueShape","page":"API","title":"ValueShapes.ConstValueShape","text":"ConstValueShape{T} <: AbstractValueShape\n\nA ConstValueShape describes the shape of constant values of type T.\n\nConstructor:\n\nConstValueShape(value)\n\nvalue may be of arbitrary type, e.g. a constant scalar value or array:\n\nConstValueShape(4.2)\nConstValueShape([11 21; 12 22])\n\nShapes of constant values have zero degrees of freedom (see totalndof).\n\nSee also the documentation of AbstractValueShape.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.NamedTupleDist","page":"API","title":"ValueShapes.NamedTupleDist","text":"NamedTupleDist <: MultivariateDistribution\nNamedTupleDist <: MultivariateDistribution\n\nA distribution with NamedTuple-typed variates.\n\nNamedTupleDist provides an effective mechanism to specify the distribution of each variable/parameter in a set of named variables/parameters.\n\nCalling varshape on a NamedTupleDist will yield a NamedTupleShape.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.NamedTupleShape","page":"API","title":"ValueShapes.NamedTupleShape","text":"NamedTupleShape{names,...} <: AbstractValueShape\n\nDefines the shape of a NamedTuple (resp.  set of variables, parameters, etc.).\n\nConstructors:\n\nNamedTupleShape(name1 = shape1::AbstractValueShape, ...)\nNamedTupleShape(named_shapes::NamedTuple)\n\nExample:\n\nshape = NamedTupleShape(\n    a = ScalarShape{Real}(),\n    b = ArrayShape{Real}(2, 3),\n    c = ConstValueShape(42)\n)\n\ndata = VectorOfSimilarVectors{Float64}(shape)\nresize!(data, 10)\nrand!(flatview(data))\ntable = shape.(data)\nfill!(table.a, 4.2)\nall(x -> x == 4.2, view(flatview(data), 1, :))\n\nSee also the documentation of AbstractValueShape.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ReshapedDist","page":"API","title":"ValueShapes.ReshapedDist","text":"ReshapedDist <: Distribution\n\nAn multivariate distribution reshaped using a given AbstractValueShape.\n\nConstructors:\n\n    ReshapedDist(dist::MultivariateDistribution, shape::AbstractValueShape)\n\nIn addition, MultivariateDistributions can be reshaped via\n\n(shape::AbstractValueShape)(dist::MultivariateDistribution)\n\nwith the difference that\n\n(shape::ArrayShape{T,1})(dist::MultivariateDistribution)\n\nwill return the original dist instead of a ReshapedDist.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ScalarShape","page":"API","title":"ValueShapes.ScalarShape","text":"ScalarShape{T} <: AbstractScalarShape{T}\n\nAn ScalarShape describes the shape of scalar values of a given type.\n\nConstructor:\n\nScalarShape{T::Type}()\n\nT may be an abstract type of Union, or a specific type, e.g.\n\nScalarShape{Real}()\nScalarShape{Integer}()\nScalarShape{Float32}()\nScalarShape{Complex}()\n\nScalar shapes may have a total number of degrees of freedom (see totalndof) greater than one, e.g. shapes of complex-valued scalars:\n\ntotalndof(ScalarShape{Real}()) == 1\ntotalndof(ScalarShape{Complex}()) == 2\n\nSee also the documentation of AbstractValueShape.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ShapedAsNT","page":"API","title":"ValueShapes.ShapedAsNT","text":"ShapedAsNT{T<:NamedTuple,...} <: AbstractArray{T,0}\n\nView of an AbstractVector{<:Real} as a zero-dimensional Array containing a NamedTuple, according to a specified NamedTupleShape.\n\nConstructors:\n\nShapedAsNT(data::AbstractVector{<:Real}, shape::NamedTupleShape)\n\nshape(data)\n\nThe resulting ShapedAsNT shares memory with data. It takes the form of a (virtual) zero-dimensional Array to make the contents as editable as data itself (compared to a standard immutable NamedTuple):\n\nx = (a = 42, b = rand(1:9, 2, 3))\nshape = valshape(x)\ndata = Vector{Int}(undef, shape)\ny = shape(data)\n@assert y isa ShapedAsNT\ny[] = x\n@assert y[] == x\ny.a = 22\ny.a[] = 33\n@assert shape(data) == y\n@assert unshaped(y) === data\n\nUse unshaped(x) to access data directly.\n\nSee also ShapedAsNTArray.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ShapedAsNTArray","page":"API","title":"ValueShapes.ShapedAsNTArray","text":"ShapedAsNTArray{T<:NamedTuple,...} <: AbstractArray{T,N}\n\nView of an AbstractArray{<:AbstractVector{<:Real},N} as an array of NamedTuples, according to a specified NamedTupleShape.\n\nShapedAsNTArray implements the Tables API. Semantically, it acts a broadcasted ShapedAsNT.\n\nConstructors:\n\nShapedAsNTArray(\n    data::AbstractArray{<:AbstractVector{<:Real},\n    shape::NamedTupleShape\n)\n\nshape.(data)\n\nThe resulting ShapedAsNTArray shares memory with data:\n\nusing ArraysOfArrays, Tables, TypedTables\n\nX = [\n    (a = 42, b = rand(1:9, 2, 3))\n    (a = 11, b = rand(1:9, 2, 3))\n]\n\nshape = valshape(X[1])\ndata = nestedview(Array{Int}(undef, totalndof(shape), 2))\nY = shape.(data)\n@assert Y isa ShapedAsNTArray\nY[:] = X\n@assert Y[1] == X[1] == shape(data[1])[]\n@assert Y.a == [42, 11]\nTables.columns(Y)\n@assert unshaped.(Y) === data\n@assert Table(Y) isa TypedTables.Table\n\nUse unshaped.(Y) to access data directly.\n\nTables.columns(Y) will return a NamedTuple of columns. They will contain a copy the data, using a memory layout as contiguous as possible for each column.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.ValueAccessor","page":"API","title":"ValueShapes.ValueAccessor","text":"ValueAccessor{S<:AbstractValueShape}\n\nA value accessor provides a means to access a value with a given shape stored in a flat real-valued data vector with a given offset position.\n\nConstructor:\n\nValueAccessor{S}(shape::S, offset::Int)\n\nThe offset is relative to the first index of a flat data array, so if the value is stored at the beginning of the array, the offset will be zero.\n\nAn ValueAccessor can be used to index into a given flat data array.\n\nExample:\n\nacc = ValueAccessor(ArrayShape{Real}(2,3), 2)\nvalshape(acc) == ArrayShape{Real,2}((2, 3))\ndata = [1, 2, 3, 4, 5, 6, 7, 8, 9]\ndata[acc] == [3 5 7; 4 6 8]\n\nNote: Subtypes of AbstractValueShape should specialize ValueShapes.vs_getindex, ValueShapes.vs_unsafe_view and ValueShapes.vs_setindex! for their ValueAccessor{...}. Specializing Base.getindex, Base.view, Base.unsafe_view or Base.setindex! directly may result in method ambiguities with custom array tapes that specialize these functions in a very generic fashion.\n\n\n\n\n\n","category":"type"},{"location":"api/#ValueShapes.const_zero","page":"API","title":"ValueShapes.const_zero","text":"const_zero(x::Any)\n\nGet the equivalent of a constant zero for values the same type as .\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.const_zero_shape-Tuple{ConstValueShape}","page":"API","title":"ValueShapes.const_zero_shape","text":"const_zero_shape(shape::ConstValueShape)\n\nGet the equivalent of a constant zero shape for shape shape.\n\n\n\n\n\n","category":"method"},{"location":"api/#ValueShapes.default_datatype","page":"API","title":"ValueShapes.default_datatype","text":"ValueShapes.default_datatype(T::Type)\n\nReturn a default specific type U that is more specific than T, with U <: T.\n\ne.g.\n\nValueShapes.default_datatype(Real) == Float64\nValueShapes.default_datatype(Complex) == Complex{Float64}\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.default_unshaped_eltype","page":"API","title":"ValueShapes.default_unshaped_eltype","text":"ValueShapes.default_unshaped_eltype(shape::AbstractValueShape)\n\nReturns the default real array element type to use for unshaped representations of data with shape shape.\n\nSubtypes of AbstractValueShape must implemenent ValueShapes.default_unshaped_eltype.\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.elshape","page":"API","title":"ValueShapes.elshape","text":"elshape(x)::AbstractValueShape\n\nGet the shape of the elements of x\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.gradient_shape","page":"API","title":"ValueShapes.gradient_shape","text":"gradient_shape(argshape::AbstractValueShape)\n\nReturn the value shape of the gradient of functions that take values of shape argshape as an input.\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.replace_const_shapes","page":"API","title":"ValueShapes.replace_const_shapes","text":"replace_const_shapes(f::Function, shape::AbstractValueShape)\n\nIf shape is a, or contains, ConstValueShape shape(s), recursively replace it/them with the result of f(s::Shape).\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.shaped_type","page":"API","title":"ValueShapes.shaped_type","text":"ValueShapes.shaped_type(shape::AbstractValueShape, ::Type{T}) where {T<:Real}\nValueShapes.shaped_type(shape::AbstractValueShape)\n\nReturns the type the will result from reshaping a real-valued vector (of element type T, if specified) with shape.\n\nSubtypes of AbstractValueShape must implement\n\nValueShapes.shaped_type(shape::AbstractValueShape, ::Type{T}) where {T<:Real}\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.stripscalar","page":"API","title":"ValueShapes.stripscalar","text":"stripscalar(x)\n\nDereference value x.\n\nIf x is a scalar-like object, like a 0-dimensional array or a Ref, stripscalar returns it's inner value. Otherwise, x is returned unchanged.\n\nUseful to strip shaped scalar-like views of their 0-dim array semantics (if present), but leave array-like views unchanged.\n\nExample:\n\ndata = [1, 2, 3]\nshape1 = NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Real}(2))\nx1 = shape1(data)\n@assert x1 isa AbstractArray{<:NamedTuple,0}\n@assert stripscalar(x) isa NamedTuple\n\nshape2 = ArrayShape{Real}(3)\nx2 = shape2(data)\n@assert x2 isa AbstractArray{Int,1}\n@assert ref(x2) isa AbstractArray{Int,1}\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.totalndof","page":"API","title":"ValueShapes.totalndof","text":"totalndof(shape::AbstractValueShape)\n\nGet the total number of degrees of freedom of values of the given shape.\n\nEquivalent to the length of a vector that would result from flattening the data into a sequence of real numbers, excluding any constant values.\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.unshaped","page":"API","title":"ValueShapes.unshaped","text":"unshaped(x)::AbstractVector{<:Real}\nunshaped(x, shape::AbstractValueShape)::AbstractVector{<:Real}\n\nRetrieve the unshaped underlying data of x, assuming x is a structured view (based on some AbstractValueShape) of a flat/unstructured real-valued data vector.\n\nIf shape is given, ensures that the shape of x is compatible with it. Specifying a shape may be necessary if the correct shape of x cannot be inferred from x, e.g. because x is assumed to have fewer degrees of freedom (because of constant components) than would be inferred from the plain value of x.\n\nExample:\n\nshape = NamedTupleShape(\n    a = ScalarShape{Real}(),\n    b = ArrayShape{Real}(2, 3)\n)\ndata = [1, 2, 3, 4, 5, 6, 7]\nx = shape(data)\n@assert unshaped(x) == data\n@assert unshaped(x.a) == view(data, 1:1)\n@assert unshaped(x.b) == view(data, 2:7)\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.unshaped-2","page":"API","title":"ValueShapes.unshaped","text":"unshaped(\n    f::Function,\n    orig_varshape::Union{Nothing,AbstractValueShape},\n    orig_valshape::Union{Nothing,AbstractValueShape} = nothing\n)\n\nReturn a function that     * Shapes it's input from a flat vector of Real using orig_varshape, if not nothing.     * Calls f with the (optionally) unshaped input.     * Unshapes the result to a flat vector of Real using orig_valshape, if not nothing.\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.unshaped-Tuple{Distributions.MultivariateDistribution{S} where S<:Distributions.ValueSupport}","page":"API","title":"ValueShapes.unshaped","text":"unshaped(d::Distributions.Distribution)\n\nTurns d into a Distributions.Distribution{Multivariate} based on varshape(d).\n\n\n\n\n\n","category":"method"},{"location":"api/#ValueShapes.valshape","page":"API","title":"ValueShapes.valshape","text":"valshape(x)::AbstractValueShape\nvalshape(acc::ValueAccessor)::AbstractValueShape\n\nGet the value shape of an arbitrary value, resp. the shape a ValueAccessor is based on, or the shape of the variates for a Distribution.\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.vardof","page":"API","title":"ValueShapes.vardof","text":"vardof(f::Function)::Integer\n\nGet the number of degrees of freedom of the input/argument of f.\n\nEquivalent to totalndof(varshape(f)) (see varshape).\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.vardof-Tuple{Distributions.Distribution}","page":"API","title":"ValueShapes.vardof","text":"vardof(d::Distributions.Distribution)\n\nGet the number of degrees of freedom of the variates of distribution d. Equivalent to totalndof(varshape(d)).\n\n\n\n\n\n","category":"method"},{"location":"api/#ValueShapes.varshape","page":"API","title":"ValueShapes.varshape","text":"varshape(f::Function)::AbstractValueShape\n\nGet the value shape of the input/argument of an unary function f. f should support call syntax\n\nf(x::T)\n\nwith valshape(x) == varshape(f) as well all\n\nf(x::AbstractVector{<:Real})\n\nwith length(eachindex(x)) == vardof(f) (see vardof).\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.varshape-Tuple{Distributions.UnivariateDistribution{S} where S<:Distributions.ValueSupport}","page":"API","title":"ValueShapes.varshape","text":"varshape(d::Distributions.Distribution)::AbstractValueShape\n\nGet the value shape of the variates of distribution d.\n\n\n\n\n\n","category":"method"},{"location":"api/#ValueShapes.vs_getindex","page":"API","title":"ValueShapes.vs_getindex","text":"ValueShapes.vs_getindex(data::AbstractArray{<:Real}, idxs::ValueAccessor...)\n\nSpecialize ValueShapes.vs_getindex instead of Base.getindex for ValueShapes.ValueAccessors, to avoid methods ambiguities with with certain custom array types.\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.vs_setindex!","page":"API","title":"ValueShapes.vs_setindex!","text":"ValueShapes.vs_setindex!(data::AbstractArray{<:Real}, v, idxs::ValueAccessor...)\n\nSpecialize ValueShapes.vs_setindex! instead of Base.setindex! or for ValueShapes.ValueAccessors, to avoid methods ambiguities with with certain custom array types.\n\n\n\n\n\n","category":"function"},{"location":"api/#ValueShapes.vs_unsafe_view","page":"API","title":"ValueShapes.vs_unsafe_view","text":"ValueShapes.vs_unsafe_view(data::AbstractArray{<:Real}, idxs::ValueAccessor...)\n\nSpecialize ValueShapes.vs_unsafe_view instead of Base.view or Base.unsafe_view for ValueShapes.ValueAccessors, to avoid methods ambiguities with with certain custom array types.\n\n\n\n\n\n","category":"function"},{"location":"LICENSE/#LICENSE","page":"LICENSE","title":"LICENSE","text":"","category":"section"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"using Markdown\nMarkdown.parse_file(joinpath(@__DIR__, \"..\", \"..\", \"LICENSE.md\"))","category":"page"},{"location":"#ValueShapes.jl","page":"Home","title":"ValueShapes.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ValueShapes provides Julia types to describe the shape of values, like scalars, arrays and structures.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Shapes provide a generic way to construct uninitialized values (e.g. multidimensional arrays) without using templates.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Shapes also act as a bridge between structured and flat data representations: Mathematical and statistical algorithms (e.g. optimizers, fitters, solvers, etc.) often represent variables/parameters as flat vectors of nameless real values. But user code will usually be more concise and readable if variables/parameters can have names (e.g. via NamedTuples) and non-scalar shapes. ValueShapes provides a duality of view between the two different data representations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"ValueShapes defines the shape of a value as the combination of it's data type (resp. element type, in the case of arrays) and the size of the value (relevant if the value is an array), e.g.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ValueShapes\n\nScalarShape{Real}()\nArrayShape{Real}(2, 3)\nConstValueShape([1 2; 3 4])","category":"page"},{"location":"","page":"Home","title":"Home","text":"Array shapes can be used to construct a compatible real-valued data vector:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Vector{Float64}(undef, ArrayShape{Real}(2, 3)) isa Vector{Float64}","category":"page"},{"location":"","page":"Home","title":"Home","text":"ValueShapes also provides a way to define the shape of a NamedTuple. This can be used to specify the names and shapes of a set of variables or parameters. Consider a fitting problem with the following parameters: A scalar a, a 2x3 array b and an array c pinned to a fixed value. This set parameters can be specified as","category":"page"},{"location":"","page":"Home","title":"Home","text":"parshapes = NamedTupleShape(\n    a = ScalarShape{Real}(),\n    b = ArrayShape{Real}(2, 3),\n    c = ConstValueShape([1 2; 3 4])\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"This set of parameters has","category":"page"},{"location":"","page":"Home","title":"Home","text":"totalndof(parshapes) == 7","category":"page"},{"location":"","page":"Home","title":"Home","text":"total degrees of freedom (the constant c does not contribute). The flat data representation for this NamedTupleShape is a vector of length 7:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Random\n\ndata = Vector{Float64}(undef, parshapes)\nsize(data) == (7,)\nrand!(data)","category":"page"},{"location":"","page":"Home","title":"Home","text":"which can again be viewed as a NamedTuple described by shape via","category":"page"},{"location":"","page":"Home","title":"Home","text":"data_as_ntuple = parshapes(data)\n\ndata_as_ntuple isa ShapedAsNT{<:NamedTuple{(:a, :b, :c)}}","category":"page"},{"location":"","page":"Home","title":"Home","text":"(See ShapedAsNT.)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note: The package EponymTuples may come in handy to define functions that take such tuples as parameters and deconstruct them, so that the variable names can be used directly inside the function body. The macro @unpack provided by the package Parameters can be used to unpack NamedTuples selectively.","category":"page"},{"location":"","page":"Home","title":"Home","text":"ValueShapes can also handle multiple values for sets of variables and is designed to compose well with ArraysOfArrays.jl and Tables.jl (and similar table packages). Broadcasting a shape over a vector of real-valued vectors will create a view that implements the Tables.jl API:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ArraysOfArrays, Tables, TypedTables\n\nmultidata = VectorOfSimilarVectors{Int}(parshapes)\nresize!(multidata, 10)\nrand!(flatview(multidata), 0:99)\n\nA = parshapes.(multidata)\nkeys(Tables.columns(A)) == (:a, :b, :c)","category":"page"},{"location":"","page":"Home","title":"Home","text":"ValueShapes supports this via specialized broadcasting. A is now a table-like view into the data (see ShapedAsNTArray), and shares memory with multidata. To create an independent NamedTuple of columns, with a contiguous memory layout for each column, use Tables.columns(A):","category":"page"},{"location":"","page":"Home","title":"Home","text":"tcols = Tables.columns(A)\nflatview(tcols.b) isa Array{Int}","category":"page"},{"location":"","page":"Home","title":"Home","text":"Constructing a TypedTables.Table, DataFrames.DataFrame or similar from A will have the same effect:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TypedTables\nflatview(Table(A).b) isa Array{Int}\n\nusing DataFrames\nflatview(DataFrame(A).b) isa Array{Int}","category":"page"}]
}
