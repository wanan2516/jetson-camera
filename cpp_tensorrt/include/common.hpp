#ifndef CAMERA_TENSORRT_COMMON_HPP
#define CAMERA_TENSORRT_COMMON_HPP

#include <iostream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "opencv2/opencv.hpp"

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error_code)                                             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;                                         \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kERROR)
        : reportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        std::cerr << msg << std::endl;
    }

private:
    nvinfer1::ILogger::Severity reportableSeverity;
};

inline int get_size_by_dims(const nvinfer1::Dims& dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline float clampf(float value, float min_value, float max_value) {
    return value > min_value ? (value < max_value ? value : max_value) : min_value;
}

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
};

struct PreParam {
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0.0f;
    float width = 0.0f;
};

#endif
