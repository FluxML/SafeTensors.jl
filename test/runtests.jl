using SafeTensors
using JSON3
using Test


const ref1d = [0, 1, 2]
const ref2d = [ 0  1  2  3  4;
         5  6  7  8  9;
        10 11 12 13 14;]
const ref3d = cat(reshape.(
([  0   1   2   3   4   5   6;
    7   8   9  10  11  12  13;
   14  15  16  17  18  19  20;
   21  22  23  24  25  26  27;
   28  29  30  31  32  33  34;],
 [ 35  36  37  38  39  40  41;
   42  43  44  45  46  47  48;
   49  50  51  52  53  54  55;
   56  57  58  59  60  61  62;
   63  64  65  66  67  68  69;],
 [ 70  71  72  73  74  75  76;
   77  78  79  80  81  82  83;
   84  85  86  87  88  89  90;
   91  92  93  94  95  96  97;
   98  99 100 101 102 103 104;]), 1, 5, 7)..., dims = 1)
const refs = (ref1d, ref2d, ref3d)

function _shape(s)
    s == "3" && return(tuple(3))
    s == "35" && return((3,5))
    s == "357" && return((3,5,7))
    error("unknown shape ",s)
end

function _type(s)
    s == "int8" && return(Int8)
    s == "uint8" && return(UInt8)
    s == "int16" && return(Int16)
    s == "int32" && return(Int32)
    s == "int64" && return(Int64)
    s == "float16" && return(Float16)
    s == "float32" && return(Float32)
    s == "float64" && return(Float64)
    s == "bool" && return(Bool)
    error("unknown type ",s)
end

function type_and_shape(s)
    t, s = split(s,"_")
    _type(t), _shape(s)
end

function check_tensor(::Type{T}, s, x) where {T<:Number}
    ref = T.(refs[length(s)])
    return(x ≈ ref)
end

function check_tensor(::Type{Bool}, s, x)
    ref = isodd.(refs[length(s)])
    return(x ≈ ref)
end

@testset "SafeTensors.jl" begin
    d = load_safetensors("model.safetensors")
    for k in keys(d)
        t, s = type_and_shape(k)
        @test check_tensor(t, s, d[k])
    end

    @testset "rust test" begin
        data = reshape(Float32[0, 3, 1, 4, 2, 5], (1,2,3))
        io = IOBuffer()
        SafeTensors.serialize(io, Dict("attn.0"=>data))
        out = take!(io)
        @test out == UInt8[
            64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100,
            116, 121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34,
            58, 91, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115,
            101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 0, 0, 0, 0, 0, 0, 128, 63,
            0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64
        ]
        @test SafeTensors.deserialize(out)["attn.0"] == data
        data = reshape(data, (1,1,2,3))
        SafeTensors.serialize(io, Dict("attn0"=>data))
        out = take!(io)
        @test out == UInt8[
            72, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116,
            121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
            91, 49, 44, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102,
            115, 101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32,
            32, 32, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0,
            160, 64
        ]
        @test SafeTensors.deserialize(out)["attn0"] == data
        data = reshape(data, (1,2,3))
        SafeTensors.serialize(io, Dict("attn.0"=>data))
        out = take!(io)
        parsed = SafeTensors.deserialize(out)
        out_buf = vec(parsed["attn.0"][:, 1, :])
        @test reinterpret(UInt8, out_buf) == UInt8[0,0,0,0,0,0,128,63,0,0,0,64]
        @test out_buf == Float32[0,1,2]
        out_buf = vec(parsed["attn.0"][:, :, 1])
        @test reinterpret(UInt8, out_buf) == UInt8[0,0,0,0,0,0,64,64]
        @test out_buf == Float32[0,3]
        serialized = b"8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}\x00\x00\x00\x00"
        loaded = SafeTensors.deserialize(serialized)
        @test collect(keys(loaded)) == ["test"]
        tensor = loaded["test"]
        @test size(tensor) == ()
        @test eltype(tensor) == Int32
        @test iszero(tensor[])
        serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        loaded = SafeTensors.deserialize(serialized)
        @test length(loaded) == 1
        @test collect(keys(loaded)) == ["test"]
        tensor = loaded["test"]
        @test size(tensor) == (2,2)
        @test eltype(tensor) == Int32
        @test iszero(tensor)
        tensors = Dict{String, SafeTensors.TensorInfo}()
        dtype = SafeTensors.F32
        shape = (2,2)
        data_offsets = (0, 16)
        for i = 1:10
            tensors["weight_$(i-1)"] = SafeTensors.TensorInfo(dtype, shape, data_offsets)
        end
        metadata = SafeTensors.HashMetadata(nothing, tensors)
        serialized = codeunits(JSON3.write(metadata))
        n = length(serialized)
        file = tempname()
        open(file, "w+") do io
            write(io, n)
            write(io, serialized)
            write(io, b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")
        end
        reloaded = read(file)
        @test_throws "Invalid Offset: " SafeTensors.deserialize(reloaded)
        serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file"
        @test_throws "Metadata Incomplete Buffer" SafeTensors.deserialize(serialized)
        serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        @test_throws "Metadata Incomplete Buffer" SafeTensors.deserialize(serialized)s
        serialized = b"<\x00\x00\x00\x00\xff\xff\xff{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        @test_throws "Header Too Large" SafeTensors.deserialize(serialized)
        serialized = b""
        @test_throws "Header Too Small" SafeTensors.deserialize(serialized)
        serialized = b"<\x00\x00\x00\x00\x00\x00\x00"
        @test_throws "Invalid Header Length" SafeTensors.deserialize(serialized)
        serialized = b"\x01\x00\x00\x00\x00\x00\x00\x00\xff"
        @test_throws "ArgumentError: invalid JSON" SafeTensors.deserialize(serialized)
        serialized = b"\x01\x00\x00\x00\x00\x00\x00\x00{"
        @test_throws "ArgumentError: invalid JSON" SafeTensors.deserialize(serialized)
        serialized = b"\x06\x00\x00\x00\x00\x00\x00\x00{}\x0D\x20\x09\x0A"
        @test iszero(length(SafeTensors.deserialize(serialized)))
        serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0, 0]}}"
        loaded = SafeTensors.deserialize(serialized)
        @test collect(keys(loaded)) == ["test"]
        tensor = loaded["test"]
        @test size(tensor) == (2,0)
        @test eltype(tensor) == Int32
        @test isempty(tensor)
        serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0, 4]}}"
        @test_throws "Tensor Invalid Info" SafeTensors.deserialize(serialized)
        serialized = b"O\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,18446744073709551614],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        @test_throws "OverflowError" SafeTensors.deserialize(serialized)
        serialized = b"N\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        @test_throws "OverflowError" SafeTensors.deserialize(serialized)
    end

    @testset "torch" begin
        for thfile in ("torch", "torch_metadata")
            file = joinpath(@__DIR__, "$(thfile).safetensors")
            jl_bytes = []
            for use_mmap in (true, false)
                torch_tensors = SafeTensors.deserialize(file; mmap = use_mmap)
                tfile = tempname()
                SafeTensors.serialize(tfile, Dict(torch_tensors), torch_tensors.metadata; mmap = use_mmap)
                jl_tensors = SafeTensors.deserialize(tfile; mmap = use_mmap)
                push!(jl_bytes, read(tfile))
                @test jl_tensors.metadata == torch_tensors.metadata
                for (name, tensor) in torch_tensors
                    @test collect(jl_tensors[name]) == collect(tensor)
                end
            end
            jl_bytes[1] == jl_bytes[2]
        end
    end
end
