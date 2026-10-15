# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ArraysOfArrays
import TypedTables
import Tables

using LinearAlgebra, FillArrays
using ChainRulesCore: Tangent, NoTangent, ZeroTangent, AbstractThunk, Thunk, ProjectTo, rrule, backing, unthunk
import Zygote, ForwardDiff


@testset "named_tuple_shape" begin
    @testset "functionality" begin
        get_y(x) = x.y

        data = VectorOfSimilarVectors(reshape(collect(1:22), 11, 2))
        ref_table = TypedTables.Table(
            a = [[1 3 5; 2 4 6], [12 14 16; 13 15 17]],
            b = [7, 18],
            c = [4.2, 4.2],
            x = Matrix[[11 21; 12 22], [11 21; 12 22]],
            y = [[8, 9, 10, 11], [19, 20, 21, 22]]
        )

        named_shapes = (
            a = ArrayShape{Real}(2, 3),
            b = ScalarShape{Real}(),
            c = ConstValueShape(4.2),
            x = ConstValueShape([11 21; 12 22]),
            y = ArrayShape{Real}(4)
        )

        shape = @inferred NamedTupleShape(;named_shapes...)
        sntshape = @inferred NamedTupleShape(ShapedAsNT; named_shapes...)
        @test @inferred(NamedTupleShape(named_shapes)) == shape

        @test @inferred(length(shape)) == 5
        @test @inferred(propertynames(shape)) == keys(shape)
        @test propertynames(shape, true) == (propertynames(shape)..., :_flatdof, :_accessors)

        @test shape == deepcopy(shape)
        @test isequal(shape, deepcopy(shape))

        @test @inferred(unshaped(shape(data[1]), shape)) == data[1]
        @test @inferred(unshaped(sntshape(data[1]), sntshape)) == data[1]

        @test shape[:y] == shape.y

        let flatdof = 0, accs = getproperty(shape, :_accessors)
            for i in 1:length(keys(shape))
                ishape = getindex(shape, i).shape
                flatdof += accs[i].len
                @test ishape == named_shapes[i]
                @test ishape == accs[i].shape
            end
            @test getproperty(shape, :_flatdof) == flatdof
        end

        @test ValueShapes.default_unshaped_eltype(NamedTupleShape(a=ScalarShape{Int}())) == Int

        @test @inferred(ValueShapes.default_unshaped_eltype(NamedTupleShape(a = ScalarShape{Int}(), b = ArrayShape{Float32}(2, 3)))) == Float32
        @test @inferred(ValueShapes.default_unshaped_eltype(shape)) == Float64

        @test @inferred(ValueShapes.shaped_type(shape)) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test @inferred(ValueShapes.shaped_type(shape, Float32)) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float32,2},Float32,Float64,Array{Int,2},Array{Float32,1}}}

        @test @inferred(get_y(shape)) === ValueShapes._accessors(shape).y
        @test @inferred(Base.propertynames(shape)) == (:a, :b, :c, :x, :y)
        @test @inferred(totalndof(shape)) == 11

        @test @inferred(shape(data[1])) == ref_table[1]
        @test @inferred(broadcast(shape, data)) == ref_table

        @test @inferred(merge((foo = 42,), shape)) == merge((foo = 42,), named_shapes)
        @test @inferred(NamedTupleShape(;shape...)) == shape
        @test @inferred(merge(shape)) === shape
        @test @inferred(merge(
            NamedTupleShape(x = ScalarShape{Real}(), y = ScalarShape{Real}()),
            NamedTupleShape(y = ArrayShape{Real}(3)),
            NamedTupleShape(z = ScalarShape{Real}(), a = ArrayShape{Real}(2, 4)),
        )) == NamedTupleShape(x = ScalarShape{Real}(), y = ArrayShape{Real}(3), z = ScalarShape{Real}(), a = ArrayShape{Real}(2, 4))

        @test typeof(@inferred(shape(undef))) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test typeof(@inferred(valshape(shape(undef)))) <: NamedTupleShape
        @test typeof(valshape(shape(undef))(undef)) == NamedTuple{(:a, :b, :c, :x, :y),Tuple{Array{Float64,2},Float64,Float64,Array{Int,2},Array{Float64,1}}}
        @test @inferred(shape(collect(1:11))) == (a = [1 3 5; 2 4 6], b = 7, c = 4.2, x = [11 21; 12 22], y = [8, 9, 10, 11])
        @test_throws ArgumentError shape(collect(1:12))

        @test valshape(shape.(push!(@inferred(VectorOfSimilarVectors{Float64}(shape)), @inferred(Vector{Float64}(undef, shape))))[1]) == valshape(shape(undef))

        let
            a = NamedTupleShape(x = ScalarShape{Int}(), y = ArrayShape{AbstractFloat}(2))
            b1 = NamedTupleShape(x = ScalarShape{Real}(), y = ArrayShape{Real}(2))
            b2 = NamedTupleShape(x = ScalarShape{Real}(), y = ArrayShape{Real}(3))
            c = NamedTupleShape(x = ScalarShape{Real}(), z = ArrayShape{Real}(2))

            @test @inferred(a <= b1) == true
            @test @inferred(a >= b1) == false
            @test @inferred(b1 >= a) == true
            @test @inferred(a <= b2) == false
            @test @inferred(a <= c) == false
        end

        @test @inferred(unshaped(sntshape(data[1]), sntshape)) === data[1]
        @test @inferred(unshaped(sntshape(data[1])[], sntshape)) == data[1]

        @testset "ValueShapes.ShapedAsNT" begin
            UA = copy(data[1])
            @test @inferred(size(@inferred(ValueShapes.ShapedAsNT(UA, shape)))) == ()
            A = ValueShapes.ShapedAsNT(UA, shape)

            @test @inferred(getproperty(A, :__internal_data) == data[1])
            @test @inferred(getproperty(A, :__internal_valshape) == valshape(A))

            @test @inferred(propertynames(A)) == (:a, :b, :c, :x, :y)
            @test propertynames(A, true) == (:a, :b, :c, :x, :y, :__internal_data, :__internal_valshape)
            @test @inferred(get_y(A)) == [8, 9, 10, 11]

            @test typeof(A.b) <: Integer

            @test @inferred(valshape(A)) === NamedTupleShape(ShapedAsNT; shape...)

            @test @inferred(realnumtype(typeof(A))) == Int

            @test @inferred(unshaped(A)) === UA
            @test @inferred(unshaped(A.a)) == view(UA, 1:6)
            @test @inferred(unshaped(A.b)) == view(UA, 7:7)
            @test @inferred(unshaped(A.y)) == view(UA, 8:11)

            @test @inferred(copy(A)) == A
            @test typeof(copy(A)) == typeof(A)

            @test @inferred((X -> (Y = copy(X); Y.a = [5 3 5; 9 4 5]; unshaped(Y)))(A)) == [5, 9, 3, 4, 5, 5, 7, 8, 9, 10, 11]
            @test @inferred((X -> (Y = copy(X); Y.b = 9; unshaped(Y)))(A)) == [1, 2, 3, 4, 5, 6, 9, 8, 9, 10, 11]
            @test @inferred((X -> (Y = copy(X); Y.c = 4.2; unshaped(Y)))(A)) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            @test_throws ArgumentError (X -> (Y = copy(X); Y.c = 4.3; unshaped(Y)))(A)
            @test @inferred((X -> (Y = copy(X); Y.y = [4, 7, 5, 6]; unshaped(Y)))(A)) == [1, 2, 3, 4, 5, 6, 7, 4, 7, 5, 6]

            x = (a = [5 3 5; 9 4 5], b = 9, c = 4.2, x = [11 21; 12 22], y = [4, 7, 5, 6])
            @test (B = copy(A); B[] = x; B[]) == x
            x_reordered = (b = 9, a = [5 3 5; 9 4 5], y = [4, 7, 5, 6], c = 4.2, x = [11 21; 12 22])
            @test (B = copy(A); B[] = x_reordered; B[]) == x
            @test_throws ArgumentError copy(A)[] = (a = [5 3 5; 9 4 5], b = 9, c = 4.2, x = [11 21; 12 23], y = [4, 7, 5, 6])

            @testset "rrules" begin
                # Base.Returns is Julia >= v1.7 only, so define:
                struct ReturnsValue{T} <: Function; value::T; end
                (f::ReturnsValue)(args...; kw...) = f.value

                vs_x = NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}(), b = ArrayShape{Real}(2), c = ScalarShape{Real}(), d = ArrayShape{Real}(2), e = ArrayShape{Real}(1), f = ArrayShape{Real}(1), g = ConstValueShape([0.4, 0.5, 0.6]))
                vs_dx = NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}(), b = ArrayShape{Real}(2), c = ScalarShape{Real}(), d = ArrayShape{Real}(2), e = ArrayShape{Real}(1), f = ArrayShape{Real}(1), g = ConstValueShape{typeof(Fill(1.0, 3)),false}(Fill(0.0, 3)))
                @test @inferred(gradient_shape(vs_x)) == vs_dx

                x_unshaped = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                x = vs_x(x_unshaped)
                dx_unshaped = [0.0, 1.2, 1.3, 1.4, 0.0, 0.0, 0.0, 0.0]
                dx = vs_dx(dx_unshaped)
                dx_contents = (__internal_data = unshaped(dx), __internal_valshape = valshape(dx))
                dx_tangent = Tangent{typeof(x),typeof(dx_contents)}(dx_contents)
                dx_nttangent = Tangent{typeof(x[])}(;ProjectTo(x)(dx_tangent)[]...)
                @test @inferred(Tangent(x, dx_unshaped)) == dx_tangent

                zdx_unshaped = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                zdx = vs_dx(zdx_unshaped)
                zdx_contents = (__internal_data = unshaped(zdx), __internal_valshape = valshape(zdx))
                zdx_tangent = Tangent{typeof(x),typeof(zdx_contents)}(zdx_contents)
                @test @inferred(Tangent(x, zdx_unshaped)) == zdx_tangent

                dy = (a = nothing, b = [1.2, 1.3], c = 1.4, d = NoTangent(), e = ZeroTangent(), f = nothing, g = [2.1, 2.2, 2.3])
                dy_unshaped = dx_unshaped
                dy_tangent = Tangent{typeof(dy),typeof(dy)}(dy)

                ref_dx_tangent(ΔΩ::AbstractThunk) = ref_dx_tangent(unthunk(ΔΩ))
                ref_dx_tangent(::Union{ZeroTangent,Nothing}) = ZeroTangent()
                ref_dx_tangent(::NoTangent) = NoTangent()
                ref_dx_tangent(::Any) = dx_tangent

                ref_ntdx_tangent(ΔΩ::AbstractThunk) = ref_ntdx_tangent(unthunk(ΔΩ))
                ref_ntdx_tangent(::Union{ZeroTangent,Nothing}) = ZeroTangent()
                ref_ntdx_tangent(::NoTangent) = NoTangent()
                ref_ntdx_tangent(ΔΩ::Any) = dx_nttangent

                @test @inferred(ProjectTo(x)(dx_tangent)) == dx
                @test @inferred(ProjectTo(x)(backing(dx_tangent))) == dx

                @test @inferred(rrule(getindex, x))[1] == x[]
                for unthunked_ΔΩ in [dy, dy_tangent, ZeroTangent(), NoTangent(), nothing]
                    for ΔΩ in [unthunked_ΔΩ, Thunk(ReturnsValue(unthunked_ΔΩ))]
                        @test @inferred(rrule(getindex, x)[2](ΔΩ)) == (NoTangent(), ProjectTo(x)(ref_dx_tangent(ΔΩ)))
                    end
                end

                @test rrule(unshaped, x)[1] == x_unshaped
                @test rrule(unshaped, x, vs_x)[1] == x_unshaped
                @test rrule(unshaped, x[], vs_x)[1] == x_unshaped
                for unthunked_ΔΩ in [dy_unshaped, ZeroTangent(), NoTangent(), nothing]
                    for ΔΩ in [unthunked_ΔΩ, Thunk(ReturnsValue(unthunked_ΔΩ))]
                        @test @inferred(rrule(unshaped, x)[2](ΔΩ)) == (NoTangent(), ProjectTo(x)(ref_dx_tangent(ΔΩ)))
                        @test @inferred(rrule(unshaped, x, vs_x)[2](ΔΩ)) == (NoTangent(), ProjectTo(x)(ref_dx_tangent(ΔΩ)), NoTangent())
                        @test @inferred(rrule(unshaped, x[], vs_x)[2](ΔΩ)) == (NoTangent(), ref_ntdx_tangent(ΔΩ), NoTangent())
                    end
                end

                ref_flatdx_tangent(ΔΩ::AbstractThunk) = ref_flatdx_tangent(unthunk(ΔΩ))
                ref_flatdx_tangent(::Union{ZeroTangent,Nothing}) = ZeroTangent()
                ref_flatdx_tangent(::NoTangent) = NoTangent()
                ref_flatdx_tangent(::Any) = dx_unshaped

                @test rrule(ShapedAsNT, x_unshaped, vs_x)[1] == x
                for unthunked_ΔΩ in [dx, dx_tangent, dx_nttangent, ZeroTangent(), NoTangent(), nothing]
                    for ΔΩ in [unthunked_ΔΩ, Thunk(ReturnsValue(unthunked_ΔΩ))]
                        @test @inferred(rrule(ShapedAsNT, x_unshaped, vs_x)[2](ΔΩ)) == (NoTangent(), ref_flatdx_tangent(ΔΩ), NoTangent())
                    end
                end
            end

            @testset "Zygote support" begin
                using Zygote

                vs = NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}(), b = ArrayShape{Real}(2))

                # ToDo: Make this work with @inferred:
                @test Zygote.gradient(x -> x[].a^2 + norm(x[].b)^2, vs([3, 4, 5])) == (gradient_shape(vs)([6, 8, 10]),)
                # ToDo: Pullbacks for getproperty:
                #@test Zygote.gradient(x_flat -> (x = vs(x_flat); norm(x.a)^2 + norm(x.b)^2), [3, 4, 5]) == ([6, 8, 10],)
                @test Zygote.gradient(x_flat -> (x = vs(x_flat); norm(x[].a)^2 + norm(x[].b)^2), [3, 4, 5])  == ([6, 8, 10],)

                foo(x::NamedTuple) = sum(map(x -> norm(x)^2, values(x)))

                # ToDo: Make this work with @inferred:
                @test Zygote.gradient(x -> foo(vs(x)[]), [3, 4, 5]) == ([6, 8, 10],)

                let
                    function foo(x)
                        vs = valshape(x)
                        ux = unshaped(x, vs)
                        x2 = vs(ux)
                        sum(x2.a) + sum(x2.b) + sum(x2.c)
                    end
                    vs = NamedTupleShape(a = ArrayShape{Real}(2), b = ConstValueShape([5, 6]), c = ArrayShape{Real}(2))
                    x = ShapedAsNT([1.1, 2.2, 3.3, 4.4], vs)
                    @test Zygote.gradient(foo, x)[1] == gradient_shape(NamedTupleShape(ShapedAsNT; vs...))([1.0, 1.0, 1.0, 1.0])
                end
            end
        end


        @testset "ValueShapes.ShapedAsNTArray" begin
            UA = Array(data)
            @test @inferred(size(@inferred(ValueShapes.ShapedAsNTArray(UA, shape)))) == (2,)
            A = ValueShapes.ShapedAsNTArray(UA, shape)

            @inferred(broadcast(identity, A)) === A

            @inferred typeof(@inferred broadcast(shape, data)) == typeof(A)
            @test shape.(data) == A
            @test @inferred(broadcast(unshaped, shape.(data))) == data
            @test all(i -> isassigned(unshaped.(similar(A)), i), eachindex(A))

            @test @inferred(propertynames(A)) == (:a, :b, :c, :x, :y)
            @test propertynames(A, true) == (:a, :b, :c, :x, :y, :__internal_data, :__internal_elshape)
            @test @inferred(get_y(A)) == [[8, 9, 10, 11], [19, 20, 21, 22]]

            @test @inferred(elshape(A)) === shape

            @test @inferred(realnumtype(typeof(A))) == Int

            @test @inferred(broadcast(unshaped, A)) === UA

            @test @inferred(A[1]) == (a = [1 3 5; 2 4 6], b = 7, c = 4.2, x = [11 21; 12 22], y = [8, 9, 10, 11])
            @test @inferred(view(A, 2)) isa ShapedAsNTArray{T,0} where T
            @test @inferred(view(A, 2)[] == A[2])
            @test @inferred(view(A, 2:2)) isa ShapedAsNTArray
            @test @inferred(view(A, 2:2) == A[2:2])

            @test @inferred(append!(copy(A), copy(A)))[3:4] == @inferred(A[1:2])
            @test @inferred(vcat(A, A))[3:4] == @inferred(A[1:2])

            @test size(@inferred similar(A)) == size(A)

            @test copy(A) == A
            @test typeof(copy(A)) == typeof(A)

            @test @inferred(TypedTables.Table(A)) == A
            @test typeof(@inferred flatview(TypedTables.Table(shape.(data)).y)) == Array{Int,2}

            A_zero() = shape.(VectorOfSimilarVectors(zeros(totalndof(shape), 2)))
            @test (B = A_zero(); B[:] = A; B) == A
            @test (B = A_zero(); B[:] = TypedTables.Table(A); B) == A

            @test unshaped.(A) == data
            let newshape = NamedTupleShape(a=ArrayShape{Real}(9,1), b=ArrayShape{Real}(1,2))
                newA = ValueShapes.ShapedAsNTArray(data, newshape)
                vecA = vec(newA)
                @test newA == vecA
            end

            let vecA = vec(A)
              @test A == vecA
            end

            @test getproperty(A, :__internal_data) == UA
            @test getproperty(A, :__internal_elshape) == shape

            @test @inferred(IndexStyle(A)) == IndexStyle(getproperty(A, :__internal_data))

            @test @inferred(axes(A)[1].stop == size(data)[1])

            @test A == copy(A)

            let B = empty(A)
                for p in propertynames(B)
                    @test @inferred(isempty(getproperty(B, p)))
                end
            end

            let B = copy(A), C = copy(A), D = copy(A)
                for i in 1:length(A)-1
                    b = pop!(B)
                    c = popfirst!(C)
                    d = splice!(D, i)
                    @test c == d
                end
                @test C == D
                @test B[end] == A[1]
                @test @inferred(length(A) - length(B)) == 1
                @test C[1] == A[end]
                @test @inferred(length(A) - length(C)) == 1
                B = empty(B)
                prepend!(B, A)
                @test B == A
                D = copy(A)
                prepend!(D, A)
                prepend!(D, A)
                prepend!(D, A)
                deleteat!(D, 1:length(A):length(D))
                for i in 1:length(D)-1
                    @test @inferred( D[i] == D[i+1])
                end
            end

            @test @inferred(Tables.istable(typeof(A))) == true
            @test @inferred(Tables.rowaccess(typeof(A))) == @inferred(Tables.rowaccess(A))
            @test @inferred(Tables.rowaccess(A)) == true
            @test @inferred(Tables.columnaccess(typeof(A))) == true
            @test @inferred(Tables.schema(A).names == propertynames(A))
            @test @inferred(Tables.rows(A)) == A

#           d = [[11 12 13; 14 15 16], [21. 22. 23.; 24. 25. 26.]]
#           nt = (a = d[1], b = d[2])
# typeof(d) <: AbstractArray{<:AbstractVector{<:Real}}
            d = [[rand(4,)] [rand(4,)] [rand(4,)]]
            nt = (a=ArrayShape{Int64, 2}((2,3)), b=ArrayShape{Float64, 2}((3,2)))

        end
    end


    @testset "examples" begin
        @test begin
            shape = NamedTupleShape(
                a = ArrayShape{Real}(2, 3),
                b = ScalarShape{Real}(),
                c = ConstValueShape(4.2),
                x = ConstValueShape([11 21; 12 22]),
                y = ArrayShape{Real}(4)
            )
            data = VectorOfSimilarVectors{Int}(shape)
            resize!(data, 10)
            rand!(flatview(data), 0:99)
            table = shape.(data)
            fill!(table.b, 42)
            all(x -> x == 42, view(flatview(data), 7, :))
        end
    end


    @testset "gradients" begin
        sntvs = NamedTupleShape(
            ShapedAsNT,
            a = ScalarShape{Real}(),
            b = ConstValueShape([4.2, 3.3]),
            c = ScalarShape{Real}(),
            d = ArrayShape{Real}(2),
            e = ScalarShape{Real}(),
            f = ArrayShape{Real}(2)
        )

        ntvs = NamedTupleShape(;sntvs...)

        f = let vs = sntvs
            v_unshaped_0 -> begin
                v_shaped_1 = vs(v_unshaped_0)
                v_unshaped_1 = unshaped(v_shaped_1)

                v_shaped_2 = vs(v_unshaped_1)
                v_unshaped_2 = unshaped(v_shaped_2, vs)

                v_shaped_3 = vs(v_unshaped_2)
                v_nt_1 = v_shaped_3[]
                v_unshaped_3 = unshaped(v_nt_1, vs)

                v_shaped_4 = vs(v_unshaped_3)
                v_unshaped_4 = unshaped(v_shaped_4, vs)
                
                v_shaped_5 = vs(v_unshaped_4)
                v_nt_2 = v_shaped_5[]

                x = v_nt_2
                sqrt(norm(x.a)^2 + norm(x.b)^2 + norm(x.d)^2 + norm(x.f)^2)
            end
        end

        g = let vs = sntvs
            v_shaped -> f(unshaped(v_shaped, vs))
        end

        for vs in (sntvs,)
            v = randn(totalndof(vs))

            @test @inferred(f(v)) isa Real
            @test ForwardDiff.gradient(f, v) isa AbstractVector{<:Real}
            grad_f_fw = ForwardDiff.gradient(f, v)
            @test @inferred(Zygote.gradient(f, v)[1]) isa AbstractVector{<:Real}
            grad_f_zg = Zygote.gradient(f, v)[1]
            @test grad_f_fw ≈ grad_f_zg

            @test @inferred(Zygote.gradient(g, vs(v))[1]) isa ShapedAsNT
            @test unshaped(Zygote.gradient(g, vs(v))[1], gradient_shape(vs)) == grad_f_zg

            @test @inferred(Zygote.gradient(g, vs(v)[])[1]) isa NamedTuple
            @test unshaped(Zygote.gradient(g, vs(v)[])[1], gradient_shape(vs)) == grad_f_zg
        end

        for vs in (sntvs, ntvs)
            X = VectorOfSimilarVectors(rand(totalndof(vs), 10))
            f_X = let vs = vs; X -> sum(norm.(norm.(values(Tables.columns((vs.(X))))))); end
            # Zygote gradients through ArrayOfSimilarArrays require ArraysOfArrays >= v1:
            if pkgversion(ArraysOfArrays) >= v"1"
                @test Zygote.gradient(f_X, X)[1] ≈ VectorOfSimilarVectors(ForwardDiff.gradient(f_X∘VectorOfSimilarVectors, flatview(X)))
            else
                @test_broken Zygote.gradient(f_X, X)[1] ≈ VectorOfSimilarVectors(ForwardDiff.gradient(f_X∘VectorOfSimilarVectors, flatview(X)))
            end
            sX = vs.(X)
            f_sX = sX -> norm(flatview(unshaped.(sX)))
            @test unshaped.(Zygote.gradient(f_sX, sX)[1], Ref(gradient_shape(vs))) ≈ VectorOfSimilarVectors(ForwardDiff.gradient(X_flat -> f_sX(vs.(VectorOfSimilarVectors(X_flat))), flatview(X)))
        end
    end


    @testset "nested named tuples" begin
        shape = NamedTupleShape(
            x = NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Real}(2), k = ConstValueShape(42)),
            c = ScalarShape{Real}(),
            d = NamedTupleShape(e = NamedTupleShape(f = ArrayShape{Real}(2)))
        )
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        nt = (x = (a = 1.0, b = [2.0, 3.0], k = 42), c = 4.0, d = (e = (f = [5.0, 6.0],),))
        data2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        nt2 = (x = (a = 10.0, b = [20.0, 30.0], k = 42), c = 40.0, d = (e = (f = [50.0, 60.0],),))

        get_xa(y) = y.x.a
        get_def(y) = y.d.e.f
        get_x(y) = y.x

        @test @inferred(totalndof(shape)) == 6
        @test @inferred(valshape(nt.d)) <= shape.d.shape
        @test @inferred(ValueShapes.shaped_type(shape)) == NamedTuple{(:x, :c, :d), Tuple{NamedTuple{(:a, :b, :k), Tuple{Float64, Vector{Float64}, Int}}, Float64, NamedTuple{(:e,), Tuple{NamedTuple{(:f,), Tuple{Vector{Float64}}}}}}}
        @test typeof(shape(undef)) == ValueShapes.shaped_type(shape)

        # Nested shapes with NamedTuple value type yield plain nested NamedTuples:
        y = @inferred shape(data)
        @test y == nt
        @test y.x isa NamedTuple
        @test y.d.e isa NamedTuple
        @test @inferred(get_xa(y)) == 1.0
        @test @inferred(get_def(y)) == [5.0, 6.0]
        @test @inferred(unshaped(nt, shape)) == data
        @test @inferred(unshaped(y, shape)) == data

        # Accessors of nested shapes: getindex yields a NamedTuple, view a ShapedAsNT:
        @test @inferred(data[shape.x]) == nt.x
        @test @inferred(view(data, shape.x)) isa ShapedAsNT
        @test view(data, shape.x)[] == nt.x
        @test unshaped(view(data, shape.d)) == view(data, 5:6)
        @test @inferred(shape.x(data)) == nt.x
        @test (v = copy(data); v[shape.x] = nt2.x; v) == [10.0, 20.0, 30.0, 4.0, 5.0, 6.0]

        # Offsets are relative to the first index of the data vector:
        odata = view([-1.0, -1.0, data...], Base.IdentityUnitRange(3:8))
        @test shape(odata) == nt
        @test view(odata, shape.d)[] == nt.d
        @test (v = view(copy(parent(odata)), axes(odata, 1)); v[shape.d] = nt2.d; parent(v)) == [-1.0, -1.0, 1.0, 2.0, 3.0, 4.0, 50.0, 60.0]

        @testset "ShapedAsNT" begin
            A = ShapedAsNT(copy(data), shape)
            @test @inferred(get_xa(A)) == 1.0
            @test @inferred(get_x(A)) == nt.x
            @test A.x isa NamedTuple
            @test A[] == nt
            @test (B = copy(A); B.x = nt2.x; unshaped(B)) == [10.0, 20.0, 30.0, 4.0, 5.0, 6.0]
            @test (B = copy(A); B.d = nt2.d; unshaped(B)) == [1.0, 2.0, 3.0, 4.0, 50.0, 60.0]
            @test (B = copy(A); B.x = view(data2, shape.x); unshaped(B)) == [10.0, 20.0, 30.0, 4.0, 5.0, 6.0]
            @test (B = copy(A); B[] = nt2; unshaped(B)) == data2
            @test_throws ArgumentError copy(A).x = (a = 10.0, b = [20.0, 30.0], k = 43)

            # Nested shapes with ShapedAsNT value type yield mutable views:
            sshape = NamedTupleShape(ShapedAsNT, x = NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}(), b = ArrayShape{Real}(2)), c = ScalarShape{Real}())
            sdata = [1.0, 2.0, 3.0, 4.0]
            sA = sshape(sdata)
            @test @inferred(get_x(sA)) isa ShapedAsNT
            @test valshape(sA.x) == NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}(), b = ArrayShape{Real}(2))
            @test unshaped(sA.x) == view(sdata, 1:3)
            sA.x.a = 10.0
            @test sdata[1] == 10.0
            @test sA[].x isa ShapedAsNT
            @test sA[] == (x = sA.x, c = 4.0)
            @test unshaped(sA[], sshape) == sdata
            @test typeof(sshape(undef)) == ValueShapes.shaped_type(sshape)
            @test sshape(undef).x isa ShapedAsNT
        end

        @testset "ShapedAsNTArray" begin
            X = VectorOfSimilarVectors(hcat(data, data2))
            XA = shape.(X)
            @test eltype(XA) == ValueShapes.shaped_type(shape)
            @test XA[1] == nt
            @test XA[2] == nt2
            @test collect(XA) == [nt, nt2]
            @test @inferred(get_x(XA)) isa ShapedAsNTArray
            @test XA.x.a == [1.0, 10.0]
            @test XA.x.b == [[2.0, 3.0], [20.0, 30.0]]
            @test XA.x.k == [42, 42]
            @test XA.c == [4.0, 40.0]
            @test XA.d.e.f == [[5.0, 6.0], [50.0, 60.0]]
            @test flatview(unshaped.(XA.x)) == flatview(X)[1:3, :]

            # Nested columns alias the data:
            XA.x.a[2] = 100.0
            @test flatview(X)[1, 2] == 100.0
            XA.x.a[2] = 10.0

            cols = Tables.columns(XA)
            @test cols.x isa ShapedAsNTArray
            @test cols.x == XA.x
            @test cols.c == [4.0, 40.0]
            cols.x.a[1] = -1.0
            @test XA.x.a[1] == 1.0
            @test Tables.schema(XA).types == (eltype(XA.x), Float64, eltype(XA.d))
            @test TypedTables.Table(XA) == XA
            sXA = NamedTupleShape(x = NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}()), c = ScalarShape{Real}()).(VectorOfSimilarVectors(ones(2, 2)))
            @test Tables.schema(sXA.x) == Tables.Schema((:a,), (Float64,))
            @test Tables.schema(sXA).types == (eltype(sXA.x), Float64)

            @test (B = copy(XA); B[1] = nt2; B[1]) == nt2
            @test (B = shape.(VectorOfSimilarVectors(zeros(6, 2))); B[:] = XA; B) == XA

            # Nested columns for other kinds of storage:
            plain = shape.([copy(data), copy(data2)])
            @test plain.x isa ShapedAsNTArray
            @test plain.x[2] == nt2.x
            @test plain.x.b == [[2.0, 3.0], [20.0, 30.0]]
            plain.x.b[1][1] = 100.0
            @test plain[1].x.b[1] == 100.0
            @test Tables.columns(plain).x isa ShapedAsNTArray
            matrix = shape.(ArrayOfSimilarVectors(reshape(repeat(data, 1, 6), 6, 2, 3)))
            @test matrix.x isa ShapedAsNTArray{<:NamedTuple,2}
            @test matrix.x.a == fill(1.0, 2, 3)
            @test matrix.d.e.f == fill([5.0, 6.0], 2, 3)
            @test Tables.columns(matrix).x == matrix.x
        end
    end
end
