# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import ValueShapes

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        ValueShapes,
        # ToDo: Resolve the default_datatype method ambiguities and the
        # view ambiguity with Indexing.jl, then enable:
        ambiguities = false,
        # Requires a registered ValueShapes version compatible with the
        # compat entries here, re-enable after the v0.12 release:
        persistent_tasks = false
    )
end # testset
