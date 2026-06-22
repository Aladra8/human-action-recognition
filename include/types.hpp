// File: feature_extractor.hpp
// Author: Baba Drammeh (Student ID: 2085440)
// Project: Human Action Recognition 

#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Action class IDs – 1-indexed to match the dataset GT files exactly.
enum ActionClass : int {
    WALKING      = 1,
    JOGGING      = 2,
    RUNNING      = 3,
    BOXING       = 4,
    HANDWAVING   = 5,
    HANDCLAPPING = 6
};

// Convert a numeric class ID to its human-readable name.
inline std::string actionToString(int cls) {
    switch (cls) {
        case 1: return "walking";
        case 2: return "jogging";
        case 3: return "running";
        case 4: return "boxing";
        case 5: return "handwaving";
        case 6: return "handclapping";
        default: return "unknown";
    }
}

// Ground truth from one .txt file:
//   <class_id>  <x_center>  <y_center>  <width>  <height>
// Coordinates may be YOLO-normalised (0–1) or absolute pixels.
struct GroundTruth {
    int   class_id = -1;
    float x_center = 0.f;
    float y_center = 0.f;
    float width    = 0.f;
    float height   = 0.f;
};

// One sequence = 40 consecutive frames + ground truth annotation.
struct Sequence {
    std::string          folder_path;
    std::string          name;
    std::vector<cv::Mat> frames;       // up to 40 BGR images
    GroundTruth          gt;
    cv::Rect             gt_bbox_abs;  // absolute-pixel bbox for frame 20
};


// 29-dimensional feature vector fed into the SVM classifier.

struct FeatureVector {
    std::vector<float> features;
};
