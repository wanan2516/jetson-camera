#ifndef CAMERA_TENSORRT_ROI_ALARM_HPP
#define CAMERA_TENSORRT_ROI_ALARM_HPP

#include <vector>

#include "data_types.hpp"

class ROIManager {
public:
    std::vector<DetectionWithROI> Apply(const std::vector<Detection>& detections) const;
};

class AlarmLogic {
public:
    FrameResult Evaluate(const std::vector<DetectionWithROI>& detections, bool prestart_mode) const;
};

#endif
