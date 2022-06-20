# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using LinearAlgebra


@testset "retshape" begin
    f1_with_rs(x) = (x = norm(x.a)^2 + norm(x.b)^2, y = vec(sum(x.a, dims = 2)))

    function ValueShapes.retshape(::typeof(f1_with_rs), argshape::NamedTupleShape)
        NamedTupleShape(
            ShapedAsNT,
            x = ScalarShape{Real}(),
            y = ArrayShape{Real}(size(argshape.a.shape)[1])
        )
    end

    f2_with_rs(x) = (x.x * sum(x.y))

    function ValueShapes.retshape(::typeof(f2_with_rs), argshape::NamedTupleShape)
        ScalarShape{Real}()
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
    y = f1_with_rs(x)

    @test valshape(y) <= @inferred(retshape(f1_with_rs, shape_x))

    y2 = (f2_with_rs ∘ f1_with_rs)(x)
    @test valshape(y2) <= @inferred(retshape(f2_with_rs ∘ f1_with_rs, shape_x))

    f_without_retshape(arg) = ""
    @test @inferred(retshape(f_without_retshape, shape_x)) == ValueShapes.UnknownReturnShape{typeof(f_without_retshape)}(shape_x)
end
