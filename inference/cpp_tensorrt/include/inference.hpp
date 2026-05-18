#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

#include "roi_alarm.hpp"
#include "trt_detector.hpp"

struct CameraInferResult {
    cv::Mat image;
    bool alarm = false;
    bool warning = false;
    SystemState system_state = SystemState::Safe;
};

class CameraTensorRTInfer {
public:
    CameraTensorRTInfer(
        const std::string& engine_path,
        const std::string& config_path = "configs/config.json",
        const std::string& roi_config_path = "configs/roi_config.json",
        bool prestart_mode = false,
        bool settle_single_frame = false
    );

    CameraInferResult Infer(const cv::Mat& input_img);

private:
    void DrawResult(
        cv::Mat& frame,
        const FrameResult& frame_result,
        const std::chrono::steady_clock::time_point& begin,
        const std::chrono::steady_clock::time_point& end
    );

    void DrawROIWarnings(
        cv::Mat& frame,
        const FrameResult& frame_result
    );

private:
    bool prestart_mode_ = false;
    bool settle_single_frame_ = false;

    InferenceConfig inference_config_;
    std::vector<ROIRule> roi_rules_;

    std::unique_ptr<TrtDetector> detector_;
    std::unique_ptr<ROIManager> roi_manager_;
    std::unique_ptr<AlarmLogic> alarm_logic_;
};