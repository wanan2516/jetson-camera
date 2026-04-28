#ifndef CAMERA_TENSORRT_ROI_ALARM_HPP
#define CAMERA_TENSORRT_ROI_ALARM_HPP

#include <string>
#include <vector>

#include "data_types.hpp"

bool LoadSafetyConfig(
    const std::string& config_path,
    const std::string& roi_fallback_path,
    InferenceConfig& inference_config,
    std::vector<ROIRule>& roi_rules,
    int& enter_frames,
    int& exit_frames
);

class ROIManager {
public:
    explicit ROIManager(std::vector<ROIRule> roi_rules = {});

    std::vector<DetectionWithROI> Apply(const std::vector<Detection>& detections, const cv::Size& image_size) const;
    void DrawROIs(cv::Mat& image) const;
    const std::vector<ROIRule>& rules() const;

private:
    std::vector<cv::Point> ResolvePolygon(const ROIRule& roi, const cv::Size& image_size) const;
    bool PointInPolygon(const cv::Point2f& point, const std::vector<cv::Point>& polygon) const;
    float BBoxOverlapRatio(const cv::Rect2f& bbox, const ROIRule& roi, const cv::Size& image_size) const;

    std::vector<ROIRule> roi_rules_;
};

class AlarmLogic {
public:
    AlarmLogic(const std::vector<ROIRule>& roi_rules = {}, int enter_frames = 3, int exit_frames = 5);

    FrameResult Evaluate(const std::vector<DetectionWithROI>& detections, bool prestart_mode);
    int enter_frames() const;
    int exit_frames() const;

private:
    void UpdateZoneCounts(const std::vector<DetectionWithROI>& detections);
    void UpdateStateMachine();

    int enter_frames_ = 3;
    int exit_frames_ = 5;
    std::vector<ZoneStatus> zone_status_;
};

#endif
