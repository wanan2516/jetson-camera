#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "opencv2/opencv.hpp"

#include "roi_alarm.hpp"
#include "trt_detector.hpp"

namespace {

std::string EscapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return escaped;
}

const char* BoolString(bool value) {
    return value ? "true" : "false";
}

void EnsureOutputDirectory(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return;
    }
    const std::string dir = path.substr(0, slash);
    if (!dir.empty()) {
        mkdir(dir.c_str(), 0755);
    }
}

int CountROIHits(const FrameResult& result) {
    int count = 0;
    for (const auto& detection : result.detections) {
        count += static_cast<int>(detection.roi_hits.size());
    }
    return count;
}

bool IsIntegerString(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    std::size_t start = value[0] == '-' ? 1 : 0;
    if (start == value.size()) {
        return false;
    }
    for (std::size_t i = start; i < value.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(value[i]))) {
            return false;
        }
    }
    return true;
}

void WriteResultJson(const std::string& path, const FrameResult& result) {
    EnsureOutputDirectory(path);
    std::ofstream out(path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"detections\": [\n";
    for (std::size_t i = 0; i < result.detections.size(); ++i) {
        const auto& item = result.detections[i];
        const auto& detection = item.detection;
        out << "    {\n";
        out << "      \"class_id\": " << detection.class_id << ",\n";
        out << "      \"class_name\": \"person\",\n";
        out << "      \"confidence\": " << detection.confidence << ",\n";
        out << "      \"bbox\": [" << detection.bbox.x << ", " << detection.bbox.y << ", "
            << detection.bbox.x + detection.bbox.width << ", " << detection.bbox.y + detection.bbox.height << "],\n";
        out << "      \"center\": [" << detection.center.x << ", " << detection.center.y << "],\n";
        out << "      \"foot_point\": [" << detection.foot_point.x << ", " << detection.foot_point.y << "],\n";
        out << "      \"roi_hits\": [\n";
        for (std::size_t j = 0; j < item.roi_hits.size(); ++j) {
            const auto& hit = item.roi_hits[j];
            out << "        {\"roi_id\": \"" << EscapeJson(hit.roi_id)
                << "\", \"roi_name\": \"" << EscapeJson(hit.roi_name)
                << "\", \"roi_type\": \"" << EscapeJson(hit.roi_type)
                << "\", \"inside\": " << BoolString(hit.inside)
                << ", \"method\": \"" << EscapeJson(hit.method) << "\"}";
            out << (j + 1 == item.roi_hits.size() ? "\n" : ",\n");
        }
        out << "      ]\n";
        out << "    }" << (i + 1 == result.detections.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"zone_summary\": {\n";
    for (std::size_t i = 0; i < result.zone_summary.size(); ++i) {
        const auto& zone = result.zone_summary[i];
        out << "    \"" << EscapeJson(zone.roi_id) << "\": {"
            << "\"roi_id\": \"" << EscapeJson(zone.roi_id)
            << "\", \"roi_name\": \"" << EscapeJson(zone.roi_name)
            << "\", \"roi_type\": \"" << EscapeJson(zone.roi_type)
            << "\", \"person_count\": " << zone.person_count
            << ", \"raw_active\": " << BoolString(zone.raw_active)
            << ", \"stable_active\": " << BoolString(zone.stable_active)
            << ", \"enter_counter\": " << zone.enter_counter
            << ", \"exit_counter\": " << zone.exit_counter
            << "}" << (i + 1 == result.zone_summary.size() ? "\n" : ",\n");
    }
    out << "  },\n";
    out << "  \"system_state\": \"" << SystemStateToString(result.system_state) << "\",\n";
    out << "  \"allow_start\": " << BoolString(result.allow_start) << ",\n";
    out << "  \"warning\": " << BoolString(result.warning) << ",\n";
    out << "  \"alarm\": " << BoolString(result.alarm) << "\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./camera_tensorrt [engine_path] [image|video|camera_index|rtsp|gstreamer] [output_image_path] [config_json] [json_output_path] [--prestart]" << std::endl;
        return -1;
    }

    const std::string engine_path = argv[1];
    const std::string input_path = argv[2];
    std::string output_path = "assets/cpp_tensorrt_result.jpg";
    std::string config_path = "configs/config.json";
    std::string json_output_path = "outputs/result.json";
    bool prestart_mode = false;

    std::vector<std::string> positional_args;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prestart") {
            prestart_mode = true;
        } else {
            positional_args.push_back(arg);
        }
    }
    if (!positional_args.empty()) {
        output_path = positional_args[0];
    }
    if (positional_args.size() >= 2) {
        config_path = positional_args[1];
    }
    if (positional_args.size() >= 3) {
        json_output_path = positional_args[2];
    }

    InferenceConfig inference_config;
    std::vector<ROIRule> roi_rules;
    int enter_frames = 3;
    int exit_frames = 5;
    if (!LoadSafetyConfig(config_path, "configs/roi_config.json", inference_config, roi_rules, enter_frames, exit_frames)) {
        std::cerr << "Failed to load ROI/alarm config" << std::endl;
        return -1;
    }

    std::cout << "config: " << config_path << std::endl;
    std::cout << "rois: " << roi_rules.size() << std::endl;
    std::cout << "enter_frames: " << enter_frames << " exit_frames: " << exit_frames << std::endl;
    std::cout << "prestart_mode: " << BoolString(prestart_mode) << std::endl;

    TrtDetector detector(engine_path);
    detector.MakePipe(true);

    ROIManager roi_manager(roi_rules);
    AlarmLogic alarm_logic(roi_rules, enter_frames, exit_frames);

    const cv::Mat still_image = cv::imread(input_path);
    const bool is_image = !still_image.empty();
    cv::VideoCapture capture;
    if (!is_image) {
        if (IsIntegerString(input_path)) {
            capture.open(std::stoi(input_path));
        } else if (input_path.find('!') != std::string::npos) {
            capture.open(input_path, cv::CAP_GSTREAMER);
        } else {
            capture.open(input_path);
        }
        if (!capture.isOpened()) {
            std::cerr << "Failed to read input: " << input_path << std::endl;
            return -1;
        }
    }

    while (true) {
        cv::Mat frame;
        if (is_image) {
            frame = still_image.clone();
        } else {
            capture >> frame;
            if (frame.empty()) {
                break;
            }
        }

        detector.CopyFromMat(frame, inference_config.input_size);

        const auto begin = std::chrono::steady_clock::now();
        detector.Infer();
        const auto end = std::chrono::steady_clock::now();

        std::vector<Detection> detections;
        detector.PostProcess(detections, inference_config);
        const auto detection_with_roi = roi_manager.Apply(detections, frame.size());

        FrameResult frame_result;
        const int settle_frames = is_image ? std::max(alarm_logic.enter_frames(), alarm_logic.exit_frames()) : 1;
        for (int i = 0; i < settle_frames; ++i) {
            frame_result = alarm_logic.Evaluate(detection_with_roi, prestart_mode);
        }

        std::cout << "detections: " << detections.size() << std::endl;
        for (const auto& detection : detections) {
            std::cout << "class=" << detection.class_id
                      << " confidence=" << detection.confidence
                      << " bbox=[" << detection.bbox.x << ", " << detection.bbox.y
                      << ", " << detection.bbox.x + detection.bbox.width
                      << ", " << detection.bbox.y + detection.bbox.height << "]"
                      << std::endl;
        }
        std::cout << "roi_hits: " << CountROIHits(frame_result) << std::endl;
        std::cout << "system_state: " << SystemStateToString(frame_result.system_state) << std::endl;
        std::cout << "allow_start: " << BoolString(frame_result.allow_start) << std::endl;
        std::cout << "warning: " << BoolString(frame_result.warning) << std::endl;
        std::cout << "alarm: " << BoolString(frame_result.alarm) << std::endl;

        roi_manager.DrawROIs(frame);
        std::vector<Detection> plain_detections;
        plain_detections.reserve(frame_result.detections.size());
        for (const auto& item : frame_result.detections) {
            plain_detections.push_back(item.detection);
        }
        detector.DrawDetections(frame, plain_detections);

        const float millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        if (millis > 0.0f) {
            cv::putText(
                frame,
                cv::format("FPS %.2f", 1000.0f / millis),
                cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 0, 255),
                2
            );
        }
        cv::putText(
            frame,
            cv::format(
                "state=%s start=%s alarm=%s",
                SystemStateToString(frame_result.system_state),
                BoolString(frame_result.allow_start),
                BoolString(frame_result.alarm)
            ),
            cv::Point(10, 45),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            frame_result.alarm ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0),
            2
        );

        if (!output_path.empty()) {
            EnsureOutputDirectory(output_path);
            cv::imwrite(output_path, frame);
            std::cout << "Saved result: " << output_path << std::endl;
        }
        if (!json_output_path.empty()) {
            WriteResultJson(json_output_path, frame_result);
            std::cout << "Saved JSON: " << json_output_path << std::endl;
        }

        if (is_image) {
            break;
        }
    }

    return 0;
}
