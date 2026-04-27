#include "roi_alarm.hpp"

std::vector<DetectionWithROI> ROIManager::Apply(const std::vector<Detection>& detections) const {
    std::vector<DetectionWithROI> result;
    result.reserve(detections.size());
    for (const auto& detection : detections) {
        result.push_back(DetectionWithROI{detection, {}});
    }
    return result;
}

FrameResult AlarmLogic::Evaluate(const std::vector<DetectionWithROI>& detections, bool prestart_mode) const {
    FrameResult result;
    result.detections = detections;
    result.system_state = SystemState::Safe;
    result.allow_start = prestart_mode && detections.empty();
    result.warning = false;
    result.alarm = false;
    return result;
}
