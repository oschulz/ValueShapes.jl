# This file is a part of ValueShapes.jl, licensed under the MIT License (MIT).


"""
    resultshape(f, vs::AbstractValueShape)

Return the shape of values returned by `f` when applied to values of shape
`vs`.

Returns `missing` if the shape of the function result cannot be determined.
"""
resultshape(f, vs::AbstractValueShape) = missing
export resultshape
