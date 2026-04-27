#include <chrono>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "roi_alarm.hpp"
#include "trt_detector.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./camera_tensorrt [engine_path] [image_or_video_path]" << std::endl;
        return -1;
    }

    const std::string engine_path = argv[1];
    const std::string input_path = argv[2];

    InferenceConfig config;
    TrtDetector detector(engine_path);
    detector.MakePipe(true);

    ROIManager roi_manager;
    AlarmLogic alarm_logic;

    cv::VideoCapture capture(input_path);
    const bool is_video = capture.isOpened();

    while (true) {
        cv::Mat frame;
        if (is_video) {
            capture >> frame;
            if (frame.empty()) {
                break;
            }
        } else {
            frame = cv::imread(input_path);
            if (frame.empty()) {
                std::cerr << "Failed to read input: " << input_path << std::endl;
                return -1;
            }
        }

        detector.CopyFromMat(frame, config.input_size);

        const auto begin = std::chrono::steady_clock::now();
        detector.Infer();
        const auto end = std::chrono::steady_clock::now();

        std::vector<Detection> detections;
        detector.PostProcess(detections, config);
        const auto detection_with_roi = roi_manager.Apply(detections);
        const auto frame_result = alarm_logic.Evaluate(detection_with_roi, false);

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

        cv::imshow("camera_tensorrt", frame);
        const int key = cv::waitKey(is_video ? 1 : 0);
        if (key == 27) {
            break;
        }
        if (!is_video) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
