module SafeTensors

using Base: Checked
using Mmap

using DLFP8Types
using BFloat16s
using JSON3
using JSON3.StructTypes

using MappedArrays: mappedarray

Base.@enum Dtype::UInt8 begin
    # Boolan type
    BOOL
    # Unsigned byte
    U8
    # Signed byte
    I8
    # FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E5M2
    # FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E4M3
    # Signed integer (16-bit)
    I16
    # Unsigned integer (16-bit)
    U16
    # Half-precision floating point
    F16
    # Brain floating point
    BF16
    # Signed integer (32-bit)
    I32
    # Unsigned integer (32-bit)
    U32
    # Floating point (32-bit)
    F32
    # Floating point (64-bit)
    F64
    # Signed integer (64-bit)
    I64
    # Unsigned integer (64-bit)
    U64
end

const typemap = Dict(
    BOOL    => Bool,
    U8      => UInt8,
    I8      => Int8,
    F8_E5M2 => Float8_E5M2,
    F8_E4M3 => Float8_E4M3FN,
    I16     => Int16,
    U16     => UInt16,
    F16     => Float16,
    BF16    => BFloat16,
    I32     => Int32,
    U32     => UInt32,
    F32     => Float32,
    F64     => Float64,
    I64     => Int64,
    U64     => UInt64,
)

tag2type(tag::Dtype) = typemap[tag]
tag2name(tag::Dtype) = Symbol(tag)

let nametagmap = Dict(v => Dtype(k) for (k, v) in Base.Enums.namemap(Dtype)),
    typetagmap = Dict(reverse(kv) for kv in typemap)
    global function name2tag(name)
        tag = get(nametagmap, Symbol(name), nothing)
        isnothing(tag) && error("Unknown Dtype: $name")
        return tag
    end
    global function type2tag(@nospecialize T)
        tag = get(typetagmap, T, nothing)
        isnothing(tag) && error("Unsupproted data type: $T")
        return tag
    end
end

StructTypes.StructType(::Type{Dtype}) = StructTypes.CustomStruct()
StructTypes.lower(x::Dtype) = tag2name(x)
StructTypes.lowertype(::Type{Dtype}) = Symbol
StructTypes.construct(::Type{Dtype}, x::Symbol) = name2tag(x)

struct TensorInfo
    dtype::Dtype
    shape::Tuple{Vararg{UInt}}
    data_offsets::NTuple{2, UInt} # rust zero-based offsets, need +1,+0 when used as index
end
StructTypes.StructType(::Type{TensorInfo}) = StructTypes.CustomStruct()
StructTypes.lower(x::TensorInfo) = (; dtype = x.dtype, shape = x.shape, data_offsets = x.data_offsets)
StructTypes.lowertype(::Type{TensorInfo}) = @NamedTuple{dtype::Dtype, shape::Vector{UInt}, data_offsets::NTuple{2, UInt}}
StructTypes.construct(::Type{TensorInfo}, x::NamedTuple) = TensorInfo(x.dtype, Tuple(x.shape), x.data_offsets)

struct HashMetadata <: AbstractDict{String, Union{Dict{String, String}, TensorInfo}}
    metadata::Union{Dict{String, String}, Nothing}
    tensors::Dict{String, TensorInfo}
end
Base.length(m::HashMetadata) = length(m.tensors) + !isnothing(m.metadata)
Base.iterate(m::HashMetadata) = isnothing(m.metadata) ? iterate(m, nothing) : (("__metadata__" => m.metadata), nothing)
Base.iterate(m::HashMetadata, state) = isnothing(state) ? iterate(m.tensors) : iterate(m.tensors, state)
function StructTypes.construct(::Type{HashMetadata}, x::Dict{String, Union{Dict{String, String}, TensorInfo}})
    metadata = get(x, "__metadata__", nothing); delete!(x, "__metadata__")
    tensors = Dict{String, TensorInfo}(x)
    return HashMetadata(metadata, tensors)
end

struct Metadata <: AbstractDict{String, TensorInfo}
    metadata::Union{Dict{String, String}, Nothing}
    tensors::Vector{TensorInfo}
    index_map::Dict{String, UInt}
end
function Metadata(
    metadata::Union{AbstractDict{String, String}, Nothing},
    tensors::AbstractVector{Pair{String, TensorInfo}}
)
    index_map = Dict{String, UInt}(); sizehint!(index_map, length(tensors))
    tensors = map(enumerate(tensors)) do (index, (k, tensor))
        index_map[k] = index
        return tensor
    end
    return Metadata(metadata, tensors, index_map)
end
Base.length(x::Metadata) = length(x.tensors)
function Base.iterate(x::Metadata, s...)
    it = iterate(x.index_map, s...)
    isnothing(it) && return nothing
    (name, index), state = it
    tensor = @inbounds x.tensors[index]
    return (name => tensor), state
end
function Base.getindex(x::Metadata, name)
    index = x.index_map[name]
    return @inbounds x.tensors[index]
end

StructTypes.StructType(::Type{Metadata}) = StructTypes.CustomStruct()
function StructTypes.lower(x::Metadata)
    metadata = x.metadata
    tensors = Dict{String, TensorInfo}(); sizehint!(tensors, length(x.tensors))
    @inbounds for (name, index) in x.index_map
        tensors[name] = x.tensors[index]
    end
    return HashMetadata(metadata, tensors)
end
StructTypes.lowertype(::Type{Metadata}) = HashMetadata
function StructTypes.construct(::Type{Metadata}, x::HashMetadata)
    metadata = x.metadata
    tensors = sort!(collect(x.tensors); by = pair -> last(pair).data_offsets)
    return Metadata(metadata, tensors)
end

function validate(metadata::Metadata)
    start = 0
    for (i, info) in enumerate(metadata.tensors)
        s, e = info.data_offsets
        if s != start || e < s
            tensor_name = something(findfirst(==(i), metadata.index_map), "no_tensor")
            error("Invalid Offset: `$tensor_name`")
        end
        start = e
        nelements = reduce(Checked.checked_mul, info.shape; init = one(UInt))
        nbytes = Checked.checked_mul(nelements, sizeof(tag2type(info.dtype)))
        if e - s != nbytes
            error("Tensor Invalid Info")
        end
    end
    return start
end

struct SafeTensor{D} <: AbstractDict{String, AbstractArray}
    metadata::Metadata
    data::D
end
getmetadata(x::SafeTensor) = getfield(x, :metadata)
Base.getproperty(x::SafeTensor, sym::Symbol) = sym == :metadata ? getmetadata(x).metadata : getfield(x, sym)
Base.length(x::SafeTensor) = length(getmetadata(x))
function Base.iterate(x::SafeTensor, s...)
    it = iterate(getmetadata(x), s...)
    isnothing(it) && return nothing
    ((name, info), state) = it
    tensor = _tensorslice(x.data, info)
    return (name => tensor), state
end
function Base.getindex(x::SafeTensor, name)
    info = getmetadata(x)[name]
    return _tensorslice(x.data, info)
end

_from_le(x) = mappedarray(ltoh, x)
function _changemaj(x, shape::NTuple{N}) where N
    perm = ntuple(i->N+1-i, Val(N))
    return PermutedDimsArray(x, perm)
end
function _tensorslice(data, info)
    T = tag2type(info.dtype)
    shape = Int.(info.shape)
    start, stop = info.data_offsets
    tensor = @inbounds _changemaj(Base.ReshapedArray(reinterpret(T, @view(data[start+0x1:stop])), reverse(shape), ()), shape)
    return _from_le(tensor)
end

const MAX_HEADER_SIZE = 100_000_000

function read_metadata(buf::AbstractVector{UInt8})
    buffer_len = length(buf)
    buffer_len < 8 && error("Header Too Small")
    n = ltoh(@inbounds reinterpret(UInt64, @view(buf[1:8]))[1])
    n > min(MAX_HEADER_SIZE, typemax(Int)) && error("Header Too Large")
    stop = Checked.checked_add(UInt(n), 0x8)
    stop > buffer_len && error("Invalid Header Length")
    metadata = @inbounds JSON3.read(@view(buf[9:Int(stop)]), Metadata)
    buffer_end = validate(metadata)
    buffer_end + 8 + n != buffer_len && error("Metadata Incomplete Buffer")
    return (n, metadata)
end

function deserialize(buffer::AbstractVector{UInt8})
    n, metadata = read_metadata(buffer)
    data = @inbounds @view buffer[n+9:end]
    return SafeTensor(metadata, data)
end

"""
    deserialize(file::AbstractString; mmap = true)

Deserialize the lazy [`SafeTensor`](@ref) object.
"""
function deserialize(file::AbstractString; mmap = true)
    if mmap
        open(io->deserialize(Mmap.mmap(io, Vector{UInt8})), file)
    else
        deserialize(read(file))
    end
end

function prepare(
    data::AbstractDict{String, <:AbstractArray},
    data_info::Union{AbstractDict{String, String}, Nothing},
)
    len = length(data)
    tensors = Vector{valtype(data)}(undef, len)
    hmetadata = Dict{String, TensorInfo}(); sizehint!(hmetadata, len)
    data = sort!(collect(data); by = kv -> (type2tag(eltype(last(kv))), first(kv)))
    offset = zero(UInt)
    for (i, (name, tensor)) in enumerate(data)
        dtype = type2tag(eltype(tensor))
        shape = size(tensor)
        n = length(reinterpret(UInt8, tensor)) % UInt
        noffset = offset + n
        info = TensorInfo(dtype, shape, (offset, noffset))
        offset = noffset
        hmetadata[name] = info
        @inbounds tensors[i] = tensor
    end
    metadata = HashMetadata(data_info, hmetadata)
    metadata_buf = IOBuffer()
    JSON3.write(metadata_buf, metadata)
    extra = 8 - mod1(metadata_buf.size, 8)
    foreach(_->write(metadata_buf, ' '), 1:extra)
    n = UInt64(metadata_buf.size)
    header_bytes = take!(metadata_buf)
    return (n, header_bytes, offset), tensors
end

function _prepare(data, data_info)
    ((n, header_bytes, offset), tensors) = prepare(data, data_info)
    header_size = length(header_bytes) % UInt + 0x8
    expected_size = header_size + offset
    return expected_size, header_size, n, header_bytes, tensors
end

@static if Base.ENDIAN_BOM == 0x04030201
    _to_le(x) = x
else
    _to_le(x) = mappedarray(htol, x)
end

function _serialize_write(io::IO, expected_size, header_size, n, header_bytes, tensors)
    ws = zero(UInt)
    ws += write(io, htol(n))
    ws += write(io, header_bytes)
    @assert ws == header_size
    for tensor in tensors
        _tensor = _to_le(collect(_changemaj(tensor, size(tensor))))
        ws += write(io, _tensor)
    end
    @assert ws == expected_size
    return
end

function _serialize_copyto!(buf::AbstractVector{UInt8}, expected_size, header_size, n, header_bytes, tensors)
    @assert length(buf) == expected_size
    copyto!(buf, 0x1, reinterpret(UInt8, [htol(n)]), 0x1, 0x8)
    copyto!(buf, 0x9, header_bytes, 0x1, header_size - 0x8)
    pos = header_size + 0x1
    for tensor in tensors
        _tensor = _to_le(collect(_changemaj(tensor, size(tensor))))
        _tensor = reinterpret(UInt8, _tensor)
        len = UInt(length(_tensor))
        copyto!(buf, pos, _tensor, 0x1, len)
        pos += len
    end
    @assert expected_size == pos - 0x1
    return
end

serialize(buf::AbstractVector{UInt8}, data::AbstractDict{String, <:AbstractArray}, data_info::Union{AbstractDict{String, String}, Nothing} = nothing) = _serialize_copyto!(buf, _prepare(data, data_info)...)
serialize(io::IO, data::AbstractDict{String, <:AbstractArray}, data_info::Union{AbstractDict{String, String}, Nothing} = nothing) = _serialize_write(io, _prepare(data, data_info)...)

"""
    serialize(
        file::AbstractString,
        data::AbstractDict{String, <:AbstractArray},
        data_info::Union{AbstractDict{String, String}, Nothing} = nothing;
        mmap = true,
    )

Serialize the `Dict` of tensors (`data`) into `file`. Optionally, some extra information can be provided as a
 `Dict{String, String}` (`data_info`).
"""
function serialize(
    file::AbstractString,
    data::AbstractDict{String, <:AbstractArray},
    data_info::Union{AbstractDict{String, String}, Nothing} = nothing;
    mmap = true,
)
    if mmap
        open(file, "w+") do io
            expected_size, header_size, n, header_bytes, tensors = _prepare(data, data_info)
            buf = Mmap.mmap(io, Vector{UInt8}, expected_size)
            _serialize_copyto!(buf, expected_size, header_size, n, header_bytes, tensors)
            Mmap.sync!(buf)
        end
    else
        open(io->serialize(io, data, data_info), file, "w+")
    end
end

"""
    load_safetensors(filename::AbstractString; mmap = true)

Eagerly load the tensors in `filename`.
"""
function load_safetensors(filename::AbstractString; mmap = true)
    safetensor = deserialize(filename; mmap)
    tensors = Dict{String, Array}(); sizehint!(tensors, length(safetensor))
    for (name, tensor) in safetensor
        tensors[name] = collect(tensor)
    end
    return tensors
end

export load_safetensors

end
