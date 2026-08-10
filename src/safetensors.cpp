#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace mk {

static DType parse_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64") return DType::I64;
    throw std::runtime_error("unsupported safetensors dtype: " + s);
}

SafetensorsFile::SafetensorsFile(const std::string& model_dir) {
    fs::path dir(model_dir);
    fs::path index = dir / "model.safetensors.index.json";
    if (fs::exists(index)) {
        std::ifstream f(index);
        json j = json::parse(f);
        std::set<std::string> shards;
        for (auto& [tensor, shard] : j.at("weight_map").items())
            shards.insert(shard.get<std::string>());
        for (auto& s : shards) map_shard((dir / s).string());
    } else if (fs::exists(dir / "model.safetensors")) {
        map_shard((dir / "model.safetensors").string());
    } else {
        throw std::runtime_error("no safetensors found in " + model_dir);
    }
}

void SafetensorsFile::map_shard(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st{};
    fstat(fd, &st);
    size_t file_len = static_cast<size_t>(st.st_size);
    void* addr = mmap(nullptr, file_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
    // Weights are read many times per token; ask the kernel to fault them in
    // eagerly rather than one page-miss at a time on first pass.
    madvise(addr, file_len, MADV_WILLNEED);
    mappings_.push_back({addr, file_len});

    const uint8_t* base = static_cast<const uint8_t*>(addr);
    uint64_t header_len;
    std::memcpy(&header_len, base, 8);
    // Compared this way round because header_len is untrusted: `8 + header_len`
    // overflows for a large value and wraps below file_len, passing the check.
    if (file_len < 8 || header_len > file_len - 8)
        throw std::runtime_error("corrupt safetensors header in " + path);

    json header = json::parse(base + 8, base + 8 + header_len);
    const uint8_t* payload = base + 8 + header_len;
    size_t payload_len = file_len - 8 - header_len;

    for (auto& [name, meta] : header.items()) {
        if (name == "__metadata__") continue;
        TensorView t;
        t.name = name;
        t.dtype = parse_dtype(meta.at("dtype").get<std::string>());
        for (auto& s : meta.at("shape")) t.shape.push_back(s.get<int64_t>());
        auto offs = meta.at("data_offsets");
        size_t begin = offs[0].get<size_t>(), end = offs[1].get<size_t>();
        if (end > payload_len || begin > end)
            throw std::runtime_error("tensor " + name + " out of bounds in " + path);
        if (end - begin != dtype_bytes(t.dtype, static_cast<size_t>(t.numel())))
            throw std::runtime_error("tensor " + name + " size mismatch");
        t.data = payload + begin;
        tensors_[name] = std::move(t);
    }
}

const TensorView& SafetensorsFile::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

SafetensorsFile::~SafetensorsFile() {
    for (auto& m : mappings_) munmap(m.addr, m.len);
}

} // namespace mk
