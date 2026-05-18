#include "inference.hpp"

#include <algorithm>
#include <stdexcept>

CameraTensorRTInfer::CameraTensorRTInfer(
    const std::string& engine_path,
    const std::string& config_path,
    const std::string& roi_config_path,
    bool prestart_mode,
    bool settle_single_frame
)
    : prestart_mode_(prestart_mode),
      settle_single_frame_(settle_single_frame) {
    int enter_frames = 3;
    int exit_frames = 5;

    bool ok = LoadSafetyConfig(
        config_path,
        roi_config_path,
        inference_config_,
        roi_rules_,
        enter_frames,
        exit_frames
    );

    if (!ok) {
        throw std::runtime_error("Failed to load ROI/alarm config");
    }

    detector_ = std::make_unique<TrtDetector>(engine_path);
    detector_->MakePipe(true);

    roi_manager_ = std::make_unique<ROIManager>(roi_rules_);

    alarm_logic_ = std::make_unique<AlarmLogic>(
        roi_rules_,
        enter_frames,
        exit_frames
    );
}

CameraInferResult CameraTensorRTInfer::Infer(const cv::Mat& input_img) {
    if (input_img.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    cv::Mat frame = input_img.clone();

    detector_->CopyFromMat(frame, inference_config_.input_size);

    auto begin = std::chrono::steady_clock::now();

    detector_->Infer();

    auto end = std::chrono::steady_clock::now();

    std::vector<Detection> detections;
    detector_->PostProcess(detections, inference_config_);

    auto detection_with_roi = roi_manager_->Apply(
        detections,
        frame.size()
    );

    FrameResult frame_result;

    int eval_times = 1;

    if (settle_single_frame_) {
        eval_times = std::max(
            alarm_logic_->enter_frames(),
            alarm_logic_->exit_frames()
        );
    }

    for (int i = 0; i < eval_times; ++i) {
        frame_result = alarm_logic_->Evaluate(
            detection_with_roi,
            prestart_mode_
        );
    }

    DrawResult(frame, frame_result, begin, end);

    CameraInferResult infer_result;
    infer_result.image = frame;

    // Python 端只接收一个报警标志位：warning_zone 或 forbidden_zone
    // 稳定触发任意一种都返回 true。
    infer_result.alarm = frame_result.warning || frame_result.alarm;
    infer_result.warning = frame_result.warning;
    infer_result.system_state = frame_result.system_state;

    return infer_result;
}


void CameraTensorRTInfer::DrawROIWarnings(
    cv::Mat& frame,
    const FrameResult& frame_result
) {
    for (const auto& item : frame_result.detections) {
        const ROIHit* risk_hit = nullptr;

        for (const auto& hit : item.roi_hits) {
            if (hit.roi_type == "warning_zone" || hit.roi_type == "forbidden_zone") {
                risk_hit = &hit;
                break;
            }
        }

        if (risk_hit == nullptr) {
            continue;
        }

        const cv::Rect2f& bbox = item.detection.bbox;
        const int x = std::max(0, static_cast<int>(bbox.x));
        const int y = std::max(18, static_cast<int>(bbox.y) - 28);

        const std::string warning_text = cv::format("WARNING");
        
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(
            warning_text,
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
            &baseline
        );

        const cv::Rect bg_rect(
            cv::Point(x, std::max(0, y - text_size.height - 4)),
            cv::Point(
                std::min(frame.cols - 1, x + text_size.width + 8),
                std::min(frame.rows - 1, y + baseline + 4)
            )
        );

        if (bg_rect.width > 0 && bg_rect.height > 0) {
            cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 255), cv::FILLED);
        }

        cv::putText(
            frame,
            warning_text,
            cv::Point(x + 4, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255, 255, 255),
            2
        );
    }
}

void CameraTensorRTInfer::DrawResult(
    cv::Mat& frame,
    const FrameResult& frame_result,
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end
) {
    roi_manager_->DrawROIs(frame);

    std::vector<Detection> plain_detections;
    plain_detections.reserve(frame_result.detections.size());

    for (const auto& item : frame_result.detections) {
        plain_detections.push_back(item.detection);
    }

    detector_->DrawDetections(frame, plain_detections);
    DrawROIWarnings(frame, frame_result);

    float millis =
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end - begin
            ).count()
        );

    // if (millis > 0.0f) {
    //     cv::putText(
    //         frame,
    //         cv::format("FPS %.2f", 1000.0f / millis),
    //         cv::Point(10, 20),
    //         cv::FONT_HERSHEY_SIMPLEX,
    //         0.6,
    //         cv::Scalar(0, 0, 255),
    //         2
    //     );
    // }

    const bool roi_alarm = frame_result.warning || frame_result.alarm;

    // cv::putText(
    //     frame,
    //     cv::format(
    //         // "state=%s start=%s alarm=%s",
    //         // SystemStateToString(frame_result.system_state),
    //         frame_result.allow_start ? "true" : "false",
    //         roi_alarm ? "true" : "false"
    //     ),
    //     cv::Point(10, 45),
    //     cv::FONT_HERSHEY_SIMPLEX,
    //     0.6,
    //     roi_alarm ? cv::Scalar(0, 0, 255)
    //               : cv::Scalar(0, 255, 0),
    //     2
    // );
}