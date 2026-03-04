// © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>

struct TensorView {
    int32_t rows;
    int32_t cols;
    const int32_t* scales_q12; // Pointer into mmap
    const int8_t* data;        // Pointer into mmap
};

class WeightsLoader {
private:
    int fd;
    size_t size;
    uint8_t* data_ptr;
    std::unordered_map<std::string, TensorView> tensors;

public:
    WeightsLoader(const std::string& filepath) {
        fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        
        struct stat sb;
        if (fstat(fd, &sb) < 0) {
            close(fd);
            throw std::runtime_error("Cannot stat file: " + filepath);
        }
        size = sb.st_size;
        
        data_ptr = (uint8_t*)mmap(nullptr, size, PROT_READ, MAP_POPULATE | MAP_PRIVATE, fd, 0);
        if (data_ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed for: " + filepath);
        }
        
        parse();
    }
    
    ~WeightsLoader() {
        if (data_ptr != MAP_FAILED) {
            munmap(data_ptr, size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
    
    TensorView get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

private:
    void parse() {
        size_t offset = 0;
        if (size < 8) throw std::runtime_error("File too small");
        
        // Check magic "DSI8" and version "1"
        if (data_ptr[0] != 'D' || data_ptr[1] != 'S' || data_ptr[2] != 'I' || data_ptr[3] != '8') {
            throw std::runtime_error("Invalid Magic bytes, expected DSI8");
        }
        offset += 4;
        
        uint32_t version = *(uint32_t*)(data_ptr + offset);
        offset += 4;
        if (version != 1) throw std::runtime_error("Unsupported version format");
        
        // Read tensors until EOF
        while (offset < size) {
            if (offset + 4 > size) break;
            uint32_t name_len = *(uint32_t*)(data_ptr + offset);
            offset += 4;
            
            std::string name((char*)(data_ptr + offset), name_len);
            offset += name_len;
            
            uint32_t rows = *(uint32_t*)(data_ptr + offset);
            offset += 4;
            uint32_t cols = *(uint32_t*)(data_ptr + offset);
            offset += 4;
            
            TensorView tv;
            tv.rows = rows;
            tv.cols = cols;
            
            // Scaler array: rows * 4 bytes
            size_t scale_size = rows * 4;
            tv.scales_q12 = (const int32_t*)(data_ptr + offset);
            offset += scale_size;
            
            // Int8 data array: rows * cols bytes
            size_t data_size = rows * cols;
            tv.data = (const int8_t*)(data_ptr + offset);
            offset += data_size;
            
            tensors[name] = tv;
            // std::cout << "Mapped: " << name << " - " << rows << "x" << cols << std::endl;
        }
    }
};
