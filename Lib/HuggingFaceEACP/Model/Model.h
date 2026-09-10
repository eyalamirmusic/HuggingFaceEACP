#pragma once

// The weights, read from HuggingFace safetensors — a JSON header naming each
// tensor's dtype, shape and byte range, followed by one raw blob — the
// model.safetensors.index.json that says which of several shards each tensor
// is in, and the config.json beside them that carries the layer counts, the
// widths and the token ids.
//
// A published format rather than a convention read off another project's
// source, and the reason nothing here needs a converter.

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Model/GemmaConfig.h>
#include <HuggingFaceEACP/Model/GemmaTensors.h>
#include <HuggingFaceEACP/Model/ModelError.h>
#include <HuggingFaceEACP/Model/ModelFiles.h>
#include <HuggingFaceEACP/Model/SafeTensors.h>
#include <HuggingFaceEACP/Model/ShardIndex.h>
#include <HuggingFaceEACP/Model/ShardedTensors.h>
#include <HuggingFaceEACP/Model/TensorLoader.h>
#include <HuggingFaceEACP/Model/TensorType.h>
