#pragma once
#include "tensor.h"
#include <map>
#include <string>
#include <vector>

namespace mk {

// mmap-backed loader for one or more .safetensors shards.
// Views stay valid for the lifetime of this object.
class SafetensorsFile {
public:
    // `path` is a model directory containing either model.safetensors or
    // model.safetensors.index.json + shards.
    explicit SafetensorsFile(const std::string& model_dir);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    const TensorView& get(const std::string& name) const;
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const std::map<std::string, TensorView>& tensors() const { return tensors_; }

private:
    void map_shard(const std::string& path);
    struct Mapping { void* addr; size_t len; };
    std::vector<Mapping> mappings_;
    std::map<std::string, TensorView> tensors_;
};

} // namespace mk
