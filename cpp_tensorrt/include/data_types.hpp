#ifndef CAMERA_TENSORRT_DATA_TYPES_HPP
#define CAMERA_TENSORRT_DATA_TYPES_HPP

#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

enum class SystemState {
    Safe,
    PrestartChecking,
    PrestartBlocked,
    Warning,
    Alarm,
};

struct Detection {
    cv::Rect2f bbox;
    int class_id = 0;
    float confidence = 0.0f;
    cv::Point2f center;
    cv::Point2f foot_point;
};

struct ROIHit {
    std::string roi_id;
    std::string roi_name;
    std::string roi_type;
    bool inside = false;
    std::string method;
};

struct DetectionWithROI {
    Detection detection;
    std::vector<ROIHit> roi_hits;
};

struct ZoneStatus {
    std::string roi_id;
    std::string roi_name;
    std::string roi_type;
    int person_count = 0;
    bool raw_active = false;
    bool stable_active = false;
    int enter_counter = 0;
    int exit_counter = 0;
};

struct FrameResult {
    std::vector<DetectionWithROI> detections;
    std::vector<ZoneStatus> zone_summary;
    SystemState system_state = SystemState::Safe;
    bool allow_start = false;
    bool warning = false;
    bool alarm = false;
};

struct InferenceConfig {
    cv::Size input_size = cv::Size(640, 640);
    float score_threshold = 0.35f;
    float iou_threshold = 0.45f;
    int topk = 100;
    int num_labels = 1;
};

#endif
