#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// BackgroundModel
//
// Computes a static background image from all 40 frames using
// pixel-wise median.  Works perfectly for KTH because the camera never moves.
class BackgroundModel {
public:
    // Build a grayscale median background from the frame set.
    static cv::Mat computeMedian(const std::vector<cv::Mat>& frames);

    // Subtract background and return a binary foreground mask.
    static cv::Mat foregroundMask(const cv::Mat& frame,
                                  const cv::Mat& background,
                                  int threshold = 28);

    // Morphological clean-up: close holes, remove noise, dilate slightly.
    static cv::Mat cleanMask(const cv::Mat& mask);
};

// HumanDetector
//
// Finds the human in the foreground mask, tracks the bounding box across
// all 40 frames with temporal smoothing, and draws the final result.
class HumanDetector {
public:
    // Find the bounding rect of the largest foreground blob.
    static cv::Rect detectBBox(const cv::Mat& fg_mask, int min_area = 400);

    // Process a full sequence: background subtract → detect → smooth.
    // Returns one bounding box per frame.
    static std::vector<cv::Rect> processSequence(const std::vector<cv::Mat>& frames,
                                                  const cv::Mat& background,
                                                  int bg_threshold = 28);

    // Exponential moving-average smoothing on a bbox sequence.
    static std::vector<cv::Rect> smoothBBoxes(const std::vector<cv::Rect>& raw,
                                               float alpha = 0.45f);

    // Draw predicted box (red) + optional GT box (green) + label on a frame copy.
    static cv::Mat drawResult(const cv::Mat& frame,
                               const cv::Rect& pred_bbox,
                               const std::string& label,
                               const cv::Rect& gt_bbox   = {},
                               cv::Scalar pred_color     = {0, 0, 255},
                               cv::Scalar gt_color       = {0, 255, 0});
};
