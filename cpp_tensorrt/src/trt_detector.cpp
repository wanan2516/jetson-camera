#include "trt_detector.hpp"

#include <algorithm>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime_api.h>

TrtDetector::TrtDetector(const std::string& engine_file_path) {
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* model_stream = new char[size];
    assert(model_stream != nullptr);
    file.read(model_stream, size);
    file.close();

    initLibNvInferPlugins(&logger_, "");
    runtime_ = nvinfer1::createInferRuntime(logger_);
    assert(runtime_ != nullptr);
    engine_ = runtime_->deserializeCudaEngine(model_stream, size);
    assert(engine_ != nullptr);
    delete[] model_stream;

    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
    cudaStreamCreate(&stream_);

    num_bindings_ = engine_->getNbIOTensors();
    for (int i = 0; i < num_bindings_; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        assert(tensor_name != nullptr);

        Binding binding;
        const nvinfer1::DataType dtype = engine_->getTensorDataType(tensor_name);
        binding.name = tensor_name;
        binding.dsize = type_to_size(dtype);
        binding.dims = engine_->getTensorShape(tensor_name);
        binding.size = get_size_by_dims(binding.dims);

        if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
            ++num_inputs_;
            input_bindings_.push_back(binding);
        } else {
            ++num_outputs_;
            output_bindings_.push_back(binding);
        }
    }
}

TrtDetector::~TrtDetector() {
    delete context_;
    delete engine_;
    delete runtime_;
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
    for (auto* ptr : device_ptrs_) {
        CHECK(cudaFree(ptr));
    }
    for (auto* ptr : host_ptrs_) {
        CHECK(cudaFreeHost(ptr));
    }
}

void TrtDetector::MakePipe(bool warmup) {
    for (const auto& binding : input_bindings_) {
        void* device_ptr = nullptr;
        CHECK(cudaMalloc(&device_ptr, binding.size * binding.dsize));
        device_ptrs_.push_back(device_ptr);
    }

    for (const auto& binding : output_bindings_) {
        void* device_ptr = nullptr;
        void* host_ptr = nullptr;
        const size_t size = binding.size * binding.dsize;
        CHECK(cudaMalloc(&device_ptr, size));
        CHECK(cudaHostAlloc(&host_ptr, size, 0));
        device_ptrs_.push_back(device_ptr);
        host_ptrs_.push_back(host_ptr);
    }

    int device_index = 0;
    for (const auto& binding : input_bindings_) {
        context_->setTensorAddress(binding.name.c_str(), device_ptrs_[device_index++]);
    }
    for (const auto& binding : output_bindings_) {
        context_->setTensorAddress(binding.name.c_str(), device_ptrs_[device_index++]);
    }

    if (!warmup) {
        return;
    }

    for (int i = 0; i < 5; ++i) {
        for (const auto& binding : input_bindings_) {
            const size_t size = binding.size * binding.dsize;
            void* host_ptr = std::malloc(size);
            std::memset(host_ptr, 0, size);
            CHECK(cudaMemcpyAsync(device_ptrs_[0], host_ptr, size, cudaMemcpyHostToDevice, stream_));
            std::free(host_ptr);
        }
        Infer();
    }
}

void TrtDetector::Letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& input_size) {
    const float input_h = static_cast<float>(input_size.height);
    const float input_w = static_cast<float>(input_size.width);
    const float height = static_cast<float>(image.rows);
    const float width = static_cast<float>(image.cols);

    const float ratio = std::min(input_h / height, input_w / width);
    const int resized_w = static_cast<int>(std::round(width * ratio));
    const int resized_h = static_cast<int>(std::round(height * ratio));

    cv::Mat resized;
    if (static_cast<int>(width) != resized_w || static_cast<int>(height) != resized_h) {
        cv::resize(image, resized, cv::Size(resized_w, resized_h));
    } else {
        resized = image.clone();
    }

    float dw = input_w - resized_w;
    float dh = input_h - resized_h;
    dw /= 2.0f;
    dh /= 2.0f;

    const int top = static_cast<int>(std::round(dh - 0.1f));
    const int bottom = static_cast<int>(std::round(dh + 0.1f));
    const int left = static_cast<int>(std::round(dw - 0.1f));
    const int right = static_cast<int>(std::round(dw + 0.1f));

    cv::copyMakeBorder(resized, resized, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
    cv::dnn::blobFromImage(resized, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

    pre_param_.ratio = 1.0f / ratio;
    pre_param_.dw = dw;
    pre_param_.dh = dh;
    pre_param_.height = height;
    pre_param_.width = width;
}

void TrtDetector::CopyFromMat(const cv::Mat& image, const cv::Size& input_size) {
    cv::Mat nchw;
    Letterbox(image, nchw, input_size);
    assert(!input_bindings_.empty());
    context_->setInputShape(input_bindings_[0].name.c_str(), nvinfer1::Dims4{1, 3, input_size.height, input_size.width});
    CHECK(cudaMemcpyAsync(device_ptrs_[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream_));
}

void TrtDetector::Infer() {
    context_->enqueueV3(stream_);
    for (int i = 0; i < num_outputs_; ++i) {
        const size_t output_size = output_bindings_[i].size * output_bindings_[i].dsize;
        CHECK(cudaMemcpyAsync(host_ptrs_[i], device_ptrs_[i + num_inputs_], output_size, cudaMemcpyDeviceToHost, stream_));
    }
    cudaStreamSynchronize(stream_);
}

void TrtDetector::PostProcess(std::vector<Detection>& detections, const InferenceConfig& config) {
    detections.clear();

    const int num_channels = output_bindings_[0].dims.d[1];
    const int num_anchors = output_bindings_[0].dims.d[2];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output(num_channels, num_anchors, CV_32F, static_cast<float*>(host_ptrs_[0]));
    output = output.t();

    for (int i = 0; i < num_anchors; ++i) {
        float* row_ptr = output.row(i).ptr<float>();
        float* box_ptr = row_ptr;
        float* score_ptr = row_ptr + 4;
        float* max_score_ptr = std::max_element(score_ptr, score_ptr + config.num_labels);
        const float score = *max_score_ptr;
        if (score <= config.score_threshold) {
            continue;
        }

        const float x = *box_ptr++ - pre_param_.dw;
        const float y = *box_ptr++ - pre_param_.dh;
        const float w = *box_ptr++;
        const float h = *box_ptr;

        const float x0 = clampf((x - 0.5f * w) * pre_param_.ratio, 0.0f, pre_param_.width);
        const float y0 = clampf((y - 0.5f * h) * pre_param_.ratio, 0.0f, pre_param_.height);
        const float x1 = clampf((x + 0.5f * w) * pre_param_.ratio, 0.0f, pre_param_.width);
        const float y1 = clampf((y + 0.5f * h) * pre_param_.ratio, 0.0f, pre_param_.height);

        boxes.emplace_back(cv::Rect(cv::Point2f(x0, y0), cv::Point2f(x1, y1)));
        scores.push_back(score);
        labels.push_back(static_cast<int>(max_score_ptr - score_ptr));
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(boxes, scores, labels, config.score_threshold, config.iou_threshold, indices);
#else
    cv::dnn::NMSBoxes(boxes, scores, config.score_threshold, config.iou_threshold, indices);
#endif

    int count = 0;
    for (int index : indices) {
        if (count >= config.topk) {
            break;
        }
        Detection detection;
        detection.bbox = cv::Rect2f(boxes[index]);
        detection.class_id = labels[index];
        detection.confidence = scores[index];
        detection.center = cv::Point2f(
            detection.bbox.x + detection.bbox.width * 0.5f,
            detection.bbox.y + detection.bbox.height * 0.5f
        );
        detection.foot_point = cv::Point2f(
            detection.bbox.x + detection.bbox.width * 0.5f,
            detection.bbox.y + detection.bbox.height
        );
        detections.push_back(detection);
        ++count;
    }
}

void TrtDetector::DrawDetections(cv::Mat& image, const std::vector<Detection>& detections) const {
    for (const auto& detection : detections) {
        cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 2);
        const std::string label = cv::format("person %.2f", detection.confidence);
        cv::putText(
            image,
            label,
            cv::Point(static_cast<int>(detection.bbox.x), std::max(10, static_cast<int>(detection.bbox.y) - 8)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1
        );
    }
}
