# This file is a part of ParameterShapes.jl, licensed under the MIT License (MIT).

"""
    macro npargs(arguments)

Macro to ease definition of functions that take a named tuple of specific
type as an argument and then deconstruct it into variables.

Example:

```julia
foo(@npargs a::Integer, b, c::AbstractVector) = ...
```

is equivalent to

```julia
foo(
    (a, b, c)::NamedTuple{
        (:a, :b, :c),
        <:Tuple{Integer,Any,AbstractVector}
    }
) = ...
```
"""
macro npargs(arguments)
    @capture(arguments, (capargs__,))

    argsyms = :(())
    ntsyms = :(())
    nttypes = :(Tuple{})

    for arg in capargs
        t = if arg isa Symbol
            push!(argsyms.args, arg)
            push!(ntsyms.args, QuoteNode(arg))
            push!(nttypes.args, :Any)
            :Any
        else
            @capture(arg, n_::t_) || error("Expected \"name::type\"")
            push!(argsyms.args, n)
            push!(ntsyms.args, QuoteNode(n))
            push!(nttypes.args, t)
        end
    end

    esc(:($argsyms::NamedTuple{$ntsyms,<:$nttypes}))
end

export @npargs
