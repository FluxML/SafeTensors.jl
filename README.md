# SafeTensors.jl


[![Build Status](https://github.com/Pevnak/SafeTensors.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Pevnak/SafeTensors.jl/actions/workflows/CI.yml?query=branch%3Amain)

This packages loads data stored in [safetensor format](https://huggingface.co/docs/safetensors/index). 
Since Python is row-major and Julia is column-major, the dimensions are permuted such the tensor has the same shape as in python, but everything is correctly ordered. This includes a performance penalty in sense that we cannot be completely copy-free.

The list of dependencies is kept minimal to `JSON3` for parsing the header.

The package does not allow to save the data.

The main function is `load_safetensors` which returns a `Dict{String,V}` where keys are names of tensors and values are tensors. An example from `runtests` is as follows
```julia
julia> using SafeTensors

julia> d = load_safetensors("model.safetensors")
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

It is also possible to load just header using unexported `load_header` as 
```julia
julia> d = SafeTensors.load_header("model.safetensors")
```


