# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using LinearAlgebra


@testset "functions" begin
    f(x) = (x = norm(x.a)^2 + norm(x.b)^2, y = vec(sum(x.a, dims = 2)))

    function ValueShapes.retshape(::typeof(f), argshape::NamedTupleShape)
        NamedTupleShape(
            ShapedAsNT,
            x = ScalarShape{Real}(),
            y = ArrayShape{Real}(size(argshape.a.shape)[1])
        )
    end

    shape_x = NamedTupleShape(
        ShapedAsNT,
        a = ArrayShape{Real}(3,2),
        b = ArrayShape{Real}(2)
    )

    output_shape = NamedTupleShape(
        ShapedAsNT,
        x = ScalarShape{Real}(),
        y = ArrayShape{Real}(3)
    )

    ux = [2, 7, 4, 6, 8, 5, 3, 1]
    x = shape_x(ux)[]
    y = f(x)

    @test valshape(y) <= @inferred(retshape(f, shape_x))

    bar(arg) = ""
    @test @inferred(retshape(bar, shape_x)) == ValueShapes.UnknownReturnShape{typeof(bar)}(shape_x)
end
