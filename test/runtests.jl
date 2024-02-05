using SafeTensors
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
end
