# This file is a part of JuliaPackageTemplate.jl, licensed under the MIT License (MIT).

using Test
using ValueShapes
import Documenter

Documenter.DocMeta.setdocmeta!(
    ValueShapes,
    :DocTestSetup,
    :(using ValueShapes);
    recursive=true,
)
Documenter.doctest(ValueShapes)
