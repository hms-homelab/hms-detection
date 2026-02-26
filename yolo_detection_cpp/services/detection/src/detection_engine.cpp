#include "detection_engine.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace hms {

// 80 COCO class names
static const char* COCO_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

DetectionEngine::DetectionEngine(const std::string& model_path, int num_classes)
    : env_(ORT_LOGGING_LEVEL_WARNING, "hms-detection")
    , num_classes_(num_classes)
{
    initClassNames();

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        // Cache input/output names
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator_);
            input_names_str_.push_back(name.get());
        }

        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_str_.push_back(name.get());
        }

        // Build const char* arrays for Run()
        for (const auto& s : input_names_str_) input_names_.push_back(s.c_str());
        for (const auto& s : output_names_str_) output_names_.push_back(s.c_str());

        // Read input dimensions from model
        auto input_shape = session_->GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (input_shape.size() == 4) {
            input_height_ = static_cast<int>(input_shape[2]);
            input_width_ = static_cast<int>(input_shape[3]);
        }

        spdlog::info("ONNX model loaded: {} (input {}x{}, {} classes)",
                      model_path, input_width_, input_height_, num_classes_);

    } catch (const Ort::Exception& e) {
        spdlog::error("Failed to load ONNX model '{}': {}", model_path, e.what());
        session_.reset();
    }
}

void DetectionEngine::initClassNames() {
    class_names_.clear();
    int count = std::min(num_classes_, static_cast<int>(sizeof(COCO_NAMES) / sizeof(COCO_NAMES[0])));
    for (int i = 0; i < count; ++i) {
        class_names_.push_back(COCO_NAMES[i]);
    }
    // Pad with "classN" if num_classes > 80
    for (int i = count; i < num_classes_; ++i) {
        class_names_.push_back("class" + std::to_string(i));
    }
}

std::vector<float> DetectionEngine::preprocess(const FrameData& frame,
                                               float& scale, float& pad_x, float& pad_y) const {
    int img_w = frame.width;
    int img_h = frame.height;

    // Letterbox: scale to fit input_width_ x input_height_ maintaining aspect ratio
    float scale_x = static_cast<float>(input_width_) / img_w;
    float scale_y = static_cast<float>(input_height_) / img_h;
    scale = std::min(scale_x, scale_y);

    int new_w = static_cast<int>(std::round(img_w * scale));
    int new_h = static_cast<int>(std::round(img_h * scale));

    pad_x = (input_width_ - new_w) / 2.0f;
    pad_y = (input_height_ - new_h) / 2.0f;

    int pad_left = static_cast<int>(std::round(pad_x));
    int pad_top = static_cast<int>(std::round(pad_y));

    // Allocate NCHW tensor: [1, 3, input_height_, input_width_]
    size_t tensor_size = static_cast<size_t>(3) * input_height_ * input_width_;
    std::vector<float> tensor(tensor_size, 114.0f / 255.0f);  // Fill with gray (normalized)

    // Resize + BGR→RGB + normalize into tensor
    for (int dst_y = 0; dst_y < new_h; ++dst_y) {
        float src_yf = dst_y / scale;
        int src_y = static_cast<int>(src_yf);
        if (src_y >= img_h) src_y = img_h - 1;

        int out_y = dst_y + pad_top;
        if (out_y < 0 || out_y >= input_height_) continue;

        for (int dst_x = 0; dst_x < new_w; ++dst_x) {
            float src_xf = dst_x / scale;
            int src_x = static_cast<int>(src_xf);
            if (src_x >= img_w) src_x = img_w - 1;

            int out_x = dst_x + pad_left;
            if (out_x < 0 || out_x >= input_width_) continue;

            // Source BGR24 pixel
            const uint8_t* px = frame.pixels.data() + src_y * frame.stride + src_x * 3;
            uint8_t b = px[0];
            uint8_t g = px[1];
            uint8_t r = px[2];

            // NCHW layout: channel * H * W + y * W + x
            size_t offset = static_cast<size_t>(out_y) * input_width_ + out_x;
            tensor[0 * input_height_ * input_width_ + offset] = r / 255.0f;  // R channel
            tensor[1 * input_height_ * input_width_ + offset] = g / 255.0f;  // G channel
            tensor[2 * input_height_ * input_width_ + offset] = b / 255.0f;  // B channel
        }
    }

    return tensor;
}

std::vector<Detection> DetectionEngine::postprocess(const float* output, int num_candidates,
                                                    float conf_threshold, float iou_threshold,
                                                    float scale, float pad_x, float pad_y,
                                                    int orig_width, int orig_height,
                                                    const std::vector<std::string>& filter_classes) const {
    // Output is [1, 84, 8400] = [batch, 4+num_classes, candidates]
    // We need to iterate over candidates (columns)
    int num_values = 4 + num_classes_;

    // Build filter set
    std::unordered_set<std::string> filter_set(filter_classes.begin(), filter_classes.end());
    bool has_filter = !filter_set.empty();

    std::vector<Detection> detections;

    for (int i = 0; i < num_candidates; ++i) {
        // Each candidate is a column: output[row * num_candidates + i]
        float cx = output[0 * num_candidates + i];
        float cy = output[1 * num_candidates + i];
        float w  = output[2 * num_candidates + i];
        float h  = output[3 * num_candidates + i];

        // Find max class score
        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < num_classes_; ++c) {
            float score = output[(4 + c) * num_candidates + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_threshold) continue;

        // Class filter
        if (has_filter && best_class >= 0 && best_class < static_cast<int>(class_names_.size())) {
            if (filter_set.find(class_names_[best_class]) == filter_set.end()) continue;
        }

        // Convert cx,cy,w,h → x1,y1,x2,y2 in 640x640 space
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Reverse letterbox: subtract padding, divide by scale
        x1 = (x1 - pad_x) / scale;
        y1 = (y1 - pad_y) / scale;
        x2 = (x2 - pad_x) / scale;
        y2 = (y2 - pad_y) / scale;

        // Clamp to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_height)));

        // Skip degenerate boxes
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;

        Detection det;
        det.class_id = best_class;
        det.class_name = (best_class >= 0 && best_class < static_cast<int>(class_names_.size()))
                         ? class_names_[best_class] : "unknown";
        det.confidence = best_score;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        detections.push_back(det);
    }

    // NMS per class
    auto keep = nms(detections, iou_threshold);
    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        result.push_back(std::move(detections[idx]));
    }

    // Sort by confidence descending
    std::sort(result.begin(), result.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    return result;
}

float DetectionEngine::iou(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

std::vector<int> DetectionEngine::nms(const std::vector<Detection>& dets, float iou_threshold) {
    if (dets.empty()) return {};

    // Group by class
    std::unordered_map<int, std::vector<int>> class_indices;
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        class_indices[dets[i].class_id].push_back(i);
    }

    std::vector<int> keep;

    for (auto& [cls, indices] : class_indices) {
        // Sort indices by confidence descending
        std::sort(indices.begin(), indices.end(),
                  [&dets](int a, int b) {
                      return dets[a].confidence > dets[b].confidence;
                  });

        std::vector<bool> suppressed(indices.size(), false);

        for (size_t i = 0; i < indices.size(); ++i) {
            if (suppressed[i]) continue;
            keep.push_back(indices[i]);

            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (suppressed[j]) continue;
                if (iou(dets[indices[i]], dets[indices[j]]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return keep;
}

std::vector<Detection> DetectionEngine::detect(const FrameData& frame,
                                               float conf_threshold,
                                               float iou_threshold,
                                               const std::vector<std::string>& filter_classes) {
    if (!session_) return {};
    if (frame.pixels.empty() || frame.width <= 0 || frame.height <= 0) return {};

    // Preprocess
    float scale, pad_x, pad_y;
    auto tensor_data = preprocess(frame, scale, pad_x, pad_y);

    // Create input tensor
    std::array<int64_t, 4> input_shape = {1, 3, input_height_, input_width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, tensor_data.data(), tensor_data.size(),
        input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size());

    // Get output tensor
    auto& output_tensor = output_tensors[0];
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const float* output_data = output_tensor.GetTensorMutableData<float>();

    // Output shape: [1, 84, 8400] → num_candidates = last dim
    int num_candidates = 0;
    if (output_shape.size() == 3) {
        num_candidates = static_cast<int>(output_shape[2]);
    } else if (output_shape.size() == 2) {
        num_candidates = static_cast<int>(output_shape[1]);
    }

    if (num_candidates == 0) return {};

    return postprocess(output_data, num_candidates,
                       conf_threshold, iou_threshold,
                       scale, pad_x, pad_y,
                       frame.width, frame.height,
                       filter_classes);
}

}  // namespace hms
