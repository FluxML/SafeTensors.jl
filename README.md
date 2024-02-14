# SafeTensors.jl


[![Build Status](https://github.com/FluxML/SafeTensors.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FluxML/SafeTensors.jl/actions/workflows/CI.yml?query=branch%3Amain)

This packages loads data stored in [safetensor format](https://huggingface.co/docs/safetensors/index).
Since Python is row-major and Julia is column-major, the dimensions are permuted such the tensor has the same shape as in python, but everything is correctly ordered. This includes a performance penalty in sense that we cannot be completely copy-free.

The main function is `load_safetensors` which returns a `Dict{String,V}` where keys are names of tensors and values are tensors. An example from `runtests` is as follows
```julia
julia> using SafeTensors

julia> d = load_safetensors("test/model.safetensors")
Dict{String, Array} with 27 entries:
  "int32_357"   => Int32[0 7 … 21 28; 35 42 … 56 63; 70 77 … 91 98;;; 1 8 … 22 29…
  "uint8_3"     => UInt8[0x00, 0x01, 0x02]
  "float16_35"  => Float16[0.0 1.0 … 3.0 4.0; 5.0 6.0 … 8.0 9.0; 10.0 11.0 … 13.0…
  "bool_3"      => Bool[0, 1, 0]
  "int64_3"     => [0, 1, 2]
  "int64_35"    => [0 1 … 3 4; 5 6 … 8 9; 10 11 … 13 14]
  "float32_357" => Float32[0.0 7.0 … 21.0 28.0; 35.0 42.0 … 56.0 63.0; 70.0 77.0 …
  "bool_35"     => Bool[0 1 … 1 0; 1 0 … 0 1; 0 1 … 1 0]
  "float32_35"  => Float32[0.0 1.0 … 3.0 4.0; 5.0 6.0 … 8.0 9.0; 10.0 11.0 … 13.0…
  "float32_3"   => Float32[0.0, 1.0, 2.0]
  "uint8_35"    => UInt8[0x00 0x01 … 0x03 0x04; 0x05 0x06 … 0x08 0x09; 0x0a 0x0b …
  "float16_3"   => Float16[0.0, 1.0, 2.0]
  "int16_357"   => Int16[0 7 … 21 28; 35 42 … 56 63; 70 77 … 91 98;;; 1 8 … 22 29…
  "int16_3"     => Int16[0, 1, 2]
  "float64_357" => [0.0 7.0 … 21.0 28.0; 35.0 42.0 … 56.0 63.0; 70.0 77.0 … 91.0 …
  "uint8_357"   => UInt8[0x00 0x07 … 0x15 0x1c; 0x23 0x2a … 0x38 0x3f; 0x46 0x4d …
  "float16_357" => Float16[0.0 7.0 … 21.0 28.0; 35.0 42.0 … 56.0 63.0; 70.0 77.0 …
  "int32_3"     => Int32[0, 1, 2]
  "int16_35"    => Int16[0 1 … 3 4; 5 6 … 8 9; 10 11 … 13 14]
  "int8_357"    => Int8[0 7 … 21 28; 35 42 … 56 63; 70 77 … 91 98;;; 1 8 … 22 29;…
  "int8_35"     => Int8[0 1 … 3 4; 5 6 … 8 9; 10 11 … 13 14]
  "bool_357"    => Bool[0 1 … 1 0; 1 0 … 0 1; 0 1 … 1 0;;; 1 0 … 0 1; 0 1 … 1 0; …
  "float64_35"  => [0.0 1.0 … 3.0 4.0; 5.0 6.0 … 8.0 9.0; 10.0 11.0 … 13.0 14.0]
  "int8_3"      => Int8[0, 1, 2]
  "int64_357"   => [0 7 … 21 28; 35 42 … 56 63; 70 77 … 91 98;;; 1 8 … 22 29; 36 …
  "int32_35"    => Int32[0 1 … 3 4; 5 6 … 8 9; 10 11 … 13 14]
  "float64_3"   => [0.0, 1.0, 2.0]
```

It can also perform a lazy loading with `SafeTensors.deserialize("model.safetensors")` which `mmap` the file and return a `Dict`-like object:
```julia
julia> tensors = SafeTensors.deserialize("test/model.safetensors"; mmap = true #= default to `true`=#);

julia> tensors["float32_35"]
3×5 mappedarray(ltoh, PermutedDimsArray(reshape(reinterpret(Float32, view(::Vector{UInt8}, 0x0000000000000ef5:0x0000000000000f30)), 5, 3), (2, 1))) with eltype Float32:
  0.0   1.0   2.0   3.0   4.0
  5.0   6.0   7.0   8.0   9.0
 10.0  11.0  12.0  13.0  14.0
```

Serialization is also supported:

```julia
julia> using Random, BFloat16s

julia> weights = Dict("W"=>randn(BFloat16, 3, 5), "b"=>rand(BFloat16, 3))
Dict{String, Array{BFloat16}} with 2 entries:
  "W" => [0.617188 0.695312 … 0.390625 -2.0; -0.65625 -0.617188 … 0.652344 0.244141; 0.226562 2.70312 … -0.174805 -0.7773…
  "b" => [0.111816, 0.566406, 0.283203]

julia> f = tempname();

julia> SafeTensors.serialize(f, weights)

julia> loaded["W"] ≈ weights["W"]
true

julia> SafeTensors.serialize(f, weights, Dict("Package"=>"SafeTensors.jl", "version"=>"1"))

julia> loaded = SafeTensors.deserialize(f);

julia> loaded.metadata
Dict{String, String} with 2 entries:
  "Package" => "SafeTensors.jl"
  "version" => "1"
```

Working with gpu:
```julia
julia> loaded["W"]
3×5 mappedarray(ltoh, PermutedDimsArray(reshape(reinterpret(BFloat16, view(::Vector{UInt8}, 0x00000000000000b9:0x00000000000000d6)), 5, 3), (2, 1))) with eltype BFloat16:
  0.542969    0.201172   1.38281    -0.255859  -1.55469
  0.172852   -0.949219   0.0561523  -1.34375   -0.206055
 -0.0854492   1.17969   -0.265625   -0.871094   2.25

julia> using CUDA; CUDA.allowscalar(false)

julia> CuArray(loaded["W"])
3×5 CuArray{BFloat16, 2, CUDA.Mem.DeviceBuffer}:
  0.542969    0.201172   1.38281    -0.255859  -1.55469
  0.172852   -0.949219   0.0561523  -1.34375   -0.206055
 -0.0854492   1.17969   -0.265625   -0.871094   2.25

julia> gpu_weights = Dict("W"=>CuArray(loaded["W"]), "b"=>CuArray(loaded["b"]))
Dict{String, CuArray{BFloat16, N, CUDA.Mem.DeviceBuffer} where N} with 2 entries:
  "W" => [0.542969 0.201172 … -0.255859 -1.55469; 0.172852 -0.949219 … -1.34375 -0.206055; -0.0854492 1.17969 … -0.871094…
  "b" => BFloat16[0.871094, 0.773438, 0.703125]

julia> f = tempname();

julia> SafeTensors.serialize(f, gpu_weights)

julia> SafeTensors.deserialize(f)
SafeTensors.SafeTensor{SubArray{UInt8, 1, Vector{UInt8}, Tuple{UnitRange{UInt64}}, true}} with 2 entries:
  "W" => BFloat16[0.542969 0.201172 … -0.255859 -1.55469; 0.172852 -0.949219 … -1.34375 -0.206055; -0.0854492 1.17969 … -…
  "b" => BFloat16[0.871094, 0.773438, 0.703125]
```
