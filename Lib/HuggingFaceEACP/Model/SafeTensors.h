#pragma once

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Model/ModelError.h>
#include <HuggingFaceEACP/Model/TensorType.h>

#include <eacp/Core/Utils/MemoryMappedFile.h>
#include <eacp/GPU/Buffer/Buffer.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace HF
{
// One tensor as the header describes it: what it is called, what it is made of,
// its shape, and where its bytes sit in the blob the header is followed by.
struct TensorInfo
{
    std::string name;
    TensorType type = TensorType::F32;
    Vector<std::int64_t> shape;
    std::uint64_t blobOffset = 0;
    std::uint64_t byteCount = 0;

    std::int64_t elementCount() const;
    int rank() const;

    // Zero past the end, so a caller checking a shape against a config reads
    // the same "not that" for a missing axis as for a zero-length one.
    std::int64_t dimension(int index) const;
};

// How a buffer's bytes are arranged, beside what `storage` says each of them
// is. Elements is every tensor a checkpoint ships: one element after another,
// nothing else in the buffer. Int8Blocks is the quantized form this tree makes
// rather than reads — the elements as signed bytes followed by one fp16 scale
// per block of thirty-two, which is Kernels/Int8Blocks.h's layout — and it is a
// second axis rather than a fourteenth TensorType because no safetensors dtype
// names it and its `storage` is honestly I8.
enum class BufferLayout
{
    Elements,
    Int8Blocks
};

// A tensor uploaded to the device, and what the buffer's elements are: F32 for
// the float buffer a kernel subscripts, F16 or BF16 for one still packed two
// sixteen-bit floats to a word, which a kernel reads through
// InputBuffer::readHalf or InputBuffer::readBFloat16, and I8 in the block
// layout above for one the loader quantized.
//
// The two travel together because nothing about a GPU::Buffer says which of
// them it holds, and binding a packed buffer where a float one is expected is
// wrong by a factor of two in every index while staying silent on both
// backends. The two packed cases are not interchangeable either: bf16 has eight
// exponent bits against fp16's five, so a bf16 buffer read through readHalf is
// not less precise, it is wrong.
struct TensorBuffer
{
    eacp::GPU::Buffer buffer;
    TensorType storage = TensorType::F32;
    BufferLayout layout = BufferLayout::Elements;

    bool isPackedHalf() const { return storage == TensorType::F16; }
    bool isPackedBFloat16() const { return storage == TensorType::BF16; }
    bool isInt8Blocks() const { return layout == BufferLayout::Int8Blocks; }
};

// A run of a tensor's elements widened to float32, for a caller walking a large
// tensor a piece at a time rather than holding all of it: `bytes` are the
// tensor's own, `first` is the element to begin at, and `destination` says how
// many to take. Exactly the widening readFloats does — F16 and BF16 both widen
// exactly — and a type no shader reads is a ModelError naming the tensor.
void widenTensorElements(const TensorInfo& tensor,
                         Span<const std::uint8_t> bytes,
                         std::int64_t first,
                         Span<float> destination);

// A safetensors file: eight bytes of little-endian header length, that many
// bytes of JSON naming every tensor, then one raw blob the offsets in that JSON
// are relative to.
//
// fromFile maps the file rather than reading it, which is what makes a 5 GB
// Gemma shard affordable: nothing is resident before the first read, and the
// pages a load touches arrive from the page cache as it touches them.
// fromBytes takes a buffer already in memory. Either way the bytes outlive
// every Span handed out, which is why this is move-only. fromView is the
// third, and the one where that is the caller's promise rather than this
// object's.
//
// Nothing here trusts the header. A file shorter than the length prefix, a
// header length that runs past the end, JSON that does not parse or is not an
// object, an entry missing dtype / shape / data_offsets, a dtype this build
// does not know, a negative or reversed byte range, a range that leaves the
// blob, and a shape whose element count disagrees with that range are each a
// ModelError rather than a read past the end of the mapping. `__metadata__` is
// not a tensor and is kept separately.
class SafeTensors
{
public:
    static SafeTensors fromFile(const std::filesystem::path& path);
    static SafeTensors fromBytes(Vector<std::uint8_t> fileBytes);

    // Bytes this does not own and does not copy. **The caller guarantees they
    // outlive this SafeTensors and every Span it hands out** — rawBytes(),
    // readFloats() and makeBuffer() all read straight out of them.
    static SafeTensors fromView(Span<const std::uint8_t> bytes);

    // Sorted by name, since the header is read out of an ordered map.
    const Vector<TensorInfo>& tensors() const;
    Vector<std::string> names() const;

    // Null when the file has no such tensor; info() is the same lookup for a
    // caller that would only turn the null back into an error.
    const TensorInfo* find(std::string_view name) const;
    bool contains(std::string_view name) const;
    const TensorInfo& info(std::string_view name) const;

    // The `__metadata__` entry, which the format defines as string to string.
    // Empty when the file carries none.
    const std::map<std::string, std::string>& metadata() const;

    Span<const std::uint8_t> rawBytes(const TensorInfo& tensor) const;
    Span<const std::uint8_t> rawBytes(std::string_view name) const;

    // Widened to float32 on the way out whatever the file holds, for a caller
    // that wants the values on the CPU. F16 and BF16 both widen exactly, so
    // widening here and widening in a shader agree bit for bit.
    Vector<float> readFloats(std::string_view name) const;
    void readFloats(std::string_view name, Span<float> destination) const;

    // F32, F16 and BF16 go to the device as they lie in the blob — the first is
    // the float buffer a kernel subscripts, the other two the packed pairs
    // readHalf and readBFloat16 read — so none of the three costs a widened
    // copy. Anything else has no shader read of its own and is widened here.
    TensorBuffer makeBuffer(std::string_view name) const;

    // The same upload, widened to F32 whatever the blob holds, for the tensors
    // bound to a program that only subscripts floats. Gemma's norm scales are
    // the ones that ask: 2048 floats apiece against a packed read that would
    // have to exist in RMSNorm, Argmax and everything else a small tensor
    // reaches.
    TensorBuffer makeFloatBuffer(std::string_view name) const;

private:
    // Which of the three constructions the bytes came from. Kept as a state
    // rather than derived from a cached Span, because two of the three would
    // have to be a Span into a member of this object and this object is moved:
    // a shard is built and emplaced into the map that caches it.
    enum class ByteSource
    {
        Owned,
        Mapped,
        Borrowed
    };

    SafeTensors() = default;

    // What both uploads fall back on when the blob is not a layout a kernel
    // reads, and what makeFloatBuffer is for a packed tensor: widened once here
    // rather than in every kernel that might meet one.
    TensorBuffer widenedBuffer(const TensorInfo& tensor) const;

    Span<const std::uint8_t> fileBytes() const;

    // The mapping's own count. A shard is routinely past what an int
    // describes, so this is the size_t one and never Span::size().
    std::uint64_t fileByteCount() const;

    void readHeader();

    ByteSource source = ByteSource::Owned;
    Vector<std::uint8_t> ownedBytes;
    Span<const std::uint8_t> borrowedBytes;
    std::optional<eacp::MemoryMappedFile> mappedFile;
    std::uint64_t blobOffset = 0;
    Vector<TensorInfo> entries;
    std::map<std::string, std::string> metadataEntries;
};
} // namespace HF
