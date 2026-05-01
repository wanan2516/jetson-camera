#include "roi_alarm.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <utility>

namespace {

std::string ReadTextFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return "";
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::size_t FindMatching(const std::string& text, std::size_t open_pos, char open_char, char close_char) {
    int depth = 0;
    bool in_string = false;
    for (std::size_t i = open_pos; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"' && (i == 0 || text[i - 1] != '\\')) {
            in_string = !in_string;
        }
        if (in_string) {
            continue;
        }
        if (c == open_char) {
            ++depth;
        } else if (c == close_char) {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }
    return std::string::npos;
}

std::string ExtractString(const std::string& text, const std::string& key, const std::string& default_value = "") {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return match[1].str();
    }
    return default_value;
}

float ExtractFloat(const std::string& text, const std::string& key, float default_value) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+)");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return std::stof(match[1].str());
    }
    return default_value;
}

int ExtractInt(const std::string& text, const std::string& key, int default_value) {
    return static_cast<int>(std::round(ExtractFloat(text, key, static_cast<float>(default_value))));
}

bool ExtractBool(const std::string& text, const std::string& key, bool default_value) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false|1|0)");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        const std::string value = match[1].str();
        return value == "true" || value == "1";
    }
    return default_value;
}

std::string ExtractSection(const std::string& text, const std::string& key, char open_char, char close_char) {
    const std::size_t key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return "";
    }
    const std::size_t open_pos = text.find(open_char, key_pos);
    if (open_pos == std::string::npos) {
        return "";
    }
    const std::size_t close_pos = FindMatching(text, open_pos, open_char, close_char);
    if (close_pos == std::string::npos) {
        return "";
    }
    return text.substr(open_pos, close_pos - open_pos + 1);
}

std::vector<std::string> ExtractObjectsFromArray(const std::string& array_text) {
    std::vector<std::string> objects;
    for (std::size_t i = 0; i < array_text.size(); ++i) {
        if (array_text[i] != '{') {
            continue;
        }
        const std::size_t end = FindMatching(array_text, i, '{', '}');
        if (end == std::string::npos) {
            break;
        }
        objects.push_back(array_text.substr(i, end - i + 1));
        i = end;
    }
    return objects;
}

std::vector<cv::Point2f> ExtractPolygon(const std::string& roi_text) {
    std::vector<cv::Point2f> polygon;
    const std::string polygon_text = ExtractSection(roi_text, "polygon", '[', ']');
    const std::regex number_pattern("[-+]?[0-9]*\\.?[0-9]+");
    std::vector<float> values;
    for (std::sregex_iterator it(polygon_text.begin(), polygon_text.end(), number_pattern), end; it != end; ++it) {
        values.push_back(std::stof((*it)[0].str()));
    }
    for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
        polygon.emplace_back(values[i], values[i + 1]);
    }
    return polygon;
}

bool ParseROIRules(const std::string& text, std::vector<ROIRule>& roi_rules) {
    const std::string rois_text = ExtractSection(text, "rois", '[', ']');
    if (rois_text.empty()) {
        return false;
    }

    for (const std::string& roi_text : ExtractObjectsFromArray(rois_text)) {
        ROIRule rule;
        rule.roi_id = ExtractString(roi_text, "roi_id");
        rule.name = ExtractString(roi_text, "name", rule.roi_id);
        rule.roi_type = ExtractString(roi_text, "roi_type");
        rule.judge_method = ExtractString(roi_text, "judge_method", "foot_point");
        rule.coordinate_mode = ExtractString(roi_text, "coordinate_mode", "absolute");
        rule.overlap_thres = ExtractFloat(roi_text, "overlap_thres", 0.2f);
        rule.enabled = ExtractBool(roi_text, "enabled", true);
        rule.polygon = ExtractPolygon(roi_text);
        if (rule.enabled && !rule.roi_id.empty() && !rule.roi_type.empty() && rule.polygon.size() >= 3) {
            roi_rules.push_back(rule);
        }
    }

    return !roi_rules.empty();
}

}  // namespace

bool LoadSafetyConfig(
    const std::string& config_path,
    const std::string& roi_fallback_path,
    InferenceConfig& inference_config,
    std::vector<ROIRule>& roi_rules,
    int& enter_frames,
    int& exit_frames
) {
    roi_rules.clear();
    const std::string config_text = ReadTextFile(config_path);
    if (config_text.empty()) {
        std::cerr << "Failed to open config: " << config_path << std::endl;
        return false;
    }

    const int imgsz = ExtractInt(config_text, "imgsz", inference_config.input_size.width);
    inference_config.input_size = cv::Size(imgsz, imgsz);
    inference_config.score_threshold = ExtractFloat(config_text, "conf_thres", inference_config.score_threshold);
    inference_config.iou_threshold = ExtractFloat(config_text, "iou_thres", inference_config.iou_threshold);
    enter_frames = ExtractInt(config_text, "enter_frames", enter_frames);
    exit_frames = ExtractInt(config_text, "exit_frames", exit_frames);

    ParseROIRules(config_text, roi_rules);
    if (roi_rules.empty() && !roi_fallback_path.empty()) {
        ParseROIRules(ReadTextFile(roi_fallback_path), roi_rules);
    }

    return !roi_rules.empty();
}

ROIManager::ROIManager(std::vector<ROIRule> roi_rules)
    : roi_rules_(std::move(roi_rules)) {}

const std::vector<ROIRule>& ROIManager::rules() const {
    return roi_rules_;
}

std::vector<cv::Point> ROIManager::ResolvePolygon(const ROIRule& roi, const cv::Size& image_size) const {
    std::vector<cv::Point> polygon;
    polygon.reserve(roi.polygon.size());
    for (const auto& point : roi.polygon) {
        if (roi.coordinate_mode == "normalized") {
            const float x = std::max(0.0f, std::min(1.0f, point.x));
            const float y = std::max(0.0f, std::min(1.0f, point.y));
            polygon.emplace_back(
                static_cast<int>(std::round(x * (image_size.width - 1))),
                static_cast<int>(std::round(y * (image_size.height - 1)))
            );
        } else {
            polygon.emplace_back(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y)));
        }
    }
    return polygon;
}

bool ROIManager::PointInPolygon(const cv::Point2f& point, const std::vector<cv::Point>& polygon) const {
    return cv::pointPolygonTest(polygon, point, false) >= 0.0;
}

float ROIManager::BBoxOverlapRatio(const cv::Rect2f& bbox, const ROIRule& roi, const cv::Size& image_size) const {
    const int x1 = std::max(0, std::min(image_size.width, static_cast<int>(std::floor(bbox.x))));
    const int y1 = std::max(0, std::min(image_size.height, static_cast<int>(std::floor(bbox.y))));
    const int x2 = std::max(0, std::min(image_size.width, static_cast<int>(std::ceil(bbox.x + bbox.width))));
    const int y2 = std::max(0, std::min(image_size.height, static_cast<int>(std::ceil(bbox.y + bbox.height))));
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    cv::Mat mask(image_size, CV_8UC1, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> polygons{ResolvePolygon(roi, image_size)};
    cv::fillPoly(mask, polygons, cv::Scalar(1));
    const cv::Mat roi_slice = mask(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    const int inter_area = cv::countNonZero(roi_slice);
    const int bbox_area = std::max((x2 - x1) * (y2 - y1), 1);
    return static_cast<float>(inter_area) / static_cast<float>(bbox_area);
}

std::vector<DetectionWithROI> ROIManager::Apply(const std::vector<Detection>& detections, const cv::Size& image_size) const {
    std::vector<DetectionWithROI> result;
    result.reserve(detections.size());

    for (const auto& detection : detections) {
        DetectionWithROI item;
        item.detection = detection;
        for (const auto& roi : roi_rules_) {
            const std::vector<cv::Point> polygon = ResolvePolygon(roi, image_size);
            bool inside = false;
            if (roi.judge_method == "foot_point") {
                inside = PointInPolygon(detection.foot_point, polygon);
            } else if (roi.judge_method == "center_point") {
                inside = PointInPolygon(detection.center, polygon);
            } else if (roi.judge_method == "overlap") {
                inside = BBoxOverlapRatio(detection.bbox, roi, image_size) >= roi.overlap_thres;
            }

            if (inside) {
                item.roi_hits.push_back(ROIHit{roi.roi_id, roi.name, roi.roi_type, true, roi.judge_method});
            }
        }
        result.push_back(item);
    }

    return result;
}

void ROIManager::DrawROIs(cv::Mat& image) const {
    cv::Mat overlay = image.clone();
    for (const auto& roi : roi_rules_) {
        const std::vector<cv::Point> polygon = ResolvePolygon(roi, image.size());
        if (polygon.empty()) {
            continue;
        }
        const cv::Scalar color = roi.roi_type == "forbidden_zone" ? cv::Scalar(0, 0, 255)
                              : roi.roi_type == "warning_zone" ? cv::Scalar(0, 255, 255)
                                                               : cv::Scalar(255, 255, 0);
        const std::vector<std::vector<cv::Point>> polygons{polygon};
        cv::fillPoly(overlay, polygons, color);
        cv::polylines(image, polygon, true, color, 2);
    }
    cv::addWeighted(overlay, 0.14, image, 0.86, 0.0, image);
}

AlarmLogic::AlarmLogic(const std::vector<ROIRule>& roi_rules, int enter_frames, int exit_frames)
    : enter_frames_(enter_frames), exit_frames_(exit_frames) {
    zone_status_.reserve(roi_rules.size());
    for (const auto& roi : roi_rules) {
        ZoneStatus status;
        status.roi_id = roi.roi_id;
        status.roi_name = roi.name;
        status.roi_type = roi.roi_type;
        zone_status_.push_back(status);
    }
}

int AlarmLogic::enter_frames() const {
    return enter_frames_;
}

int AlarmLogic::exit_frames() const {
    return exit_frames_;
}

void AlarmLogic::UpdateZoneCounts(const std::vector<DetectionWithROI>& detections) {
    for (auto& zone : zone_status_) {
        zone.person_count = 0;
    }

    for (const auto& detection : detections) {
        for (const auto& hit : detection.roi_hits) {
            for (auto& zone : zone_status_) {
                if (zone.roi_id == hit.roi_id) {
                    zone.person_count += 1;
                    break;
                }
            }
        }
    }

    for (auto& zone : zone_status_) {
        zone.raw_active = zone.person_count > 0;
    }
}

void AlarmLogic::UpdateStateMachine() {
    for (auto& zone : zone_status_) {
        if (zone.raw_active) {
            zone.enter_counter += 1;
            zone.exit_counter = 0;
            if (zone.enter_counter >= enter_frames_) {
                zone.stable_active = true;
            }
        } else {
            zone.exit_counter += 1;
            zone.enter_counter = 0;
            if (zone.exit_counter >= exit_frames_) {
                zone.stable_active = false;
            }
        }
    }
}

FrameResult AlarmLogic::Evaluate(const std::vector<DetectionWithROI>& detections, bool prestart_mode) {
    UpdateZoneCounts(detections);
    UpdateStateMachine();

    bool clear_active = false;
    bool has_clear_zone = false;
    bool clear_confirmed = false;
    bool warning_active = false;
    bool forbidden_active = false;

    for (const auto& zone : zone_status_) {
        if (zone.roi_type == "clear_zone") {
            has_clear_zone = true;
            clear_active = clear_active || zone.stable_active;
        } else if (zone.roi_type == "warning_zone") {
            warning_active = warning_active || zone.stable_active;
        } else if (zone.roi_type == "forbidden_zone") {
            forbidden_active = forbidden_active || zone.stable_active;
        }
    }

    clear_confirmed = has_clear_zone;
    for (const auto& zone : zone_status_) {
        if (zone.roi_type == "clear_zone" && (zone.stable_active || zone.exit_counter < exit_frames_)) {
            clear_confirmed = false;
        }
    }

    FrameResult result;
    result.detections = detections;
    result.zone_summary = zone_status_;
    result.alarm = forbidden_active;
    result.warning = warning_active;

    if (result.alarm) {
        result.system_state = SystemState::Alarm;
    } else if (prestart_mode && clear_active) {
        result.system_state = SystemState::PrestartBlocked;
    } else if (result.warning) {
        result.system_state = SystemState::Warning;
    } else if (prestart_mode && !clear_confirmed) {
        result.system_state = SystemState::PrestartChecking;
    } else {
        result.system_state = SystemState::Safe;
    }

    result.allow_start = prestart_mode && clear_confirmed && !result.warning && !result.alarm;
    return result;
}
