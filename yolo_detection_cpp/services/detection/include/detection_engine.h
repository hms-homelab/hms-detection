#pragma once

#include "frame_data.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace hms {

struct Detection {
    std::string class_name;
    int class_id = -1;
    float confidence = 0.0f;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // bbox in original image coordinates
};

class DetectionEngine {
public:
    explicit DetectionEngine(const std::string& model_path, int num_classes = 80, bool gpu_enabled = false);

    /// Load the ONNX session onto GPU/CPU. Safe to call multiple times.
    void load();

    /// Unload the ONNX session, releasing GPU memory. Safe to call multiple times.
    void unload();

    /// Run inference on a single BGR24 frame
    std::vector<Detection> detect(const FrameData& frame,
                                  float conf_threshold = 0.5f,
                                  float iou_threshold = 0.45f,
                                  const std::vector<std::string>& filter_classes = {});

    const std::vector<std::string>& classNames() const { return class_names_; }
    bool isLoaded() const { std::lock_guard lock(session_mutex_); return session_ != nullptr; }
    bool isModelValid() const { return model_valid_; }
    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }

    // Exposed for testing
    std::vector<float> preprocess(const FrameData& frame,
                                  float& scale, float& pad_x, float& pad_y) const;

    std::vector<Detection> postprocess(const float* output, int num_candidates,
                                       float conf_threshold, float iou_threshold,
                                       float scale, float pad_x, float pad_y,
                                       int orig_width, int orig_height,
                                       const std::vector<std::string>& filter_classes) const;

    /// Post-process end-to-end model output [1, N, 6] where each row is [x1, y1, x2, y2, conf, class_id]
    std::vector<Detection> postprocessE2E(const float* output, int num_detections,
                                          float conf_threshold,
                                          float scale, float pad_x, float pad_y,
                                          int orig_width, int orig_height,
                                          const std::vector<std::string>& filter_classes) const;

    static std::vector<int> nms(const std::vector<Detection>& dets, float iou_threshold);
    static float iou(const Detection& a, const Detection& b);

private:
    void initClassNames();

    Ort::Env env_;
    mutable std::mutex session_mutex_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string model_path_;
    bool gpu_enabled_ = false;
    bool model_valid_ = false;  // true if model file exists and loaded successfully at least once

    std::vector<std::string> class_names_;
    int num_classes_;
    int input_width_ = 640;
    int input_height_ = 640;

    // Cached input/output names
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Log output format once on first detection
    mutable bool e2e_logged_ = false;
    mutable bool raw_logged_ = false;
    mutable bool format_error_logged_ = false;
};

}  // namespace hms
