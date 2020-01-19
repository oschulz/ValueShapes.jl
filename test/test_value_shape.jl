# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).

using ValueShapes
using Test

using Random
using ElasticArrays
import TypedTables


@testset "abstract_value_shape" begin
    @testset "default_datatype" begin
        @test @inferred(ValueShapes.default_datatype(Integer)) == Int
        @test @inferred(ValueShapes.default_datatype(AbstractFloat)) == Float64
        @test @inferred(ValueShapes.default_datatype(Real)) == Float64
        @test @inferred(ValueShapes.default_datatype(Complex)) == Complex{Float64}

        @test @inferred(elshape(Complex(1.0, 2.0))) == ScalarShape{Complex{Float64}}()
        @test @inferred(elshape([[3, 5], [3, 2]])) == ArrayShape{Int,1}((2,))
        
        arrshape = ArrayShape{Real, 2}((2,3))
        @test @inferred(length(Vector{Real}(undef, arrshape))) == 6
        @test @inferred(size(Vector{Real}(undef, arrshape))) == (6,)
        
        data = [1;2;3;4;7;8;9]
        scalarshape = ScalarShape{Real}()
        ntshape = NamedTupleShape(a=arrshape, b=scalarshape)
        shapedasnt = ntshape(data)       
        @test stripscalar(Ref(shapedasnt)) == Ref(shapedasnt)[]

        @test ValueShapes._checkcompat_inner
        
 
    end
end
