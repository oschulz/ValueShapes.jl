# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


# The ChainRulesCore extension provides an rrule for _adignore_call, so AD
# will ignore code wrapped in @_adignore:

@inline _adignore_call(f) = f()

macro _adignore(expr)
    :(_adignore_call(() -> $(esc(expr))))
end
