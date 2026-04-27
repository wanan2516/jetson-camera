#ifndef CAMERA_TENSORRT_TRT_DETECTOR_HPP
#define CAMERA_TENSORRT_TRT_DETECTOR_HPP

#include <fstream>
#include <string>
#include <vector>

#include "NvInferPlugin.h"
#include "common.hpp"
#include "data_types.hpp"

class TrtDetector {
public:
    explicit TrtDetector(const std::string& engine_file_path);
    ~TrtDetector();

    void MakePipe(bool warmup = true);
    void CopyFromMat(const cv::Mat& image, const cv::Size& input_size);
    void Infer();
    void PostProcess(std::vector<Detection>& detections, const InferenceConfig& config);
    void DrawDetections(cv::Mat& image, const std::vector<Detection>& detections) const;

private:
    void Letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& input_size);

    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    Logger logger_{nvinfer1::ILogger::Severity::kERROR};

    int num_bindings_ = 0;
    int num_inputs_ = 0;
    int num_outputs_ = 0;
    std::vector<Binding> input_bindings_;
    std::vector<Binding> output_bindings_;
    std::vector<void*> host_ptrs_;
    std::vector<void*> device_ptrs_;
    PreParam pre_param_;
};

#endif
