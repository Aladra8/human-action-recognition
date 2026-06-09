#pragma once
#include "types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

// FeatureExtractor
//
// Produces a 29-dimensional classical CV feature vector from one sequence.
//
// Feature groups:
//   f0  – f7   Centroid velocity & displacement    (who is moving and how fast)
//   f8  – f19  Dense Farneback optical flow stats  (motion intensity & direction)
//   f20 – f24  Bounding-box shape statistics       (aspect ratio, area)
//   f25 – f28  FFT-based stride frequency          (periodic leg/arm motion)
class FeatureExtractor {
public:
    // Feature layout (34 total) 
    // f0  –f7   Centroid velocity & displacement          (locomotion speed)
    // f8  –f19  Dense Farneback optical flow stats        (motion direction)
    // f20 –f24  Bounding-box shape statistics             (aspect ratio, area)
    // f25 –f28  FFT stride frequency                      (periodic motion)
    // f29 –f33  Discriminative extras                     (waving/clap/boxing)
    //   f29  Upper-body left/right flow asymmetry    (low=boxing/clap, high=wave)
    //   f30  Net upward flow in upper body            (high=waving, low=boxing)
    //   f31  Temporal std of upper-body flow mag      (variable=wave, steady=box)
    //   f32  Speed acceleration std                   (run>jog>walk)
    //   f33  Max foreground coverage (normalised)     (scale invariance)
    static constexpr int FEATURE_DIM = 34;

    static FeatureVector extract(const std::vector<cv::Mat>& frames,
                                  const std::vector<cv::Rect>& bboxes);

private:
    static void velocityFeatures  (const std::vector<cv::Rect>& bboxes,
                                    int img_w, std::vector<float>& out);
    static void flowFeatures      (const std::vector<cv::Mat>& frames,
                                    const std::vector<cv::Rect>& bboxes,
                                    std::vector<float>& out);
    static void shapeFeatures     (const std::vector<cv::Rect>& bboxes,
                                    int img_w, int img_h,
                                    std::vector<float>& out);
    static void frequencyFeatures (const std::vector<cv::Rect>& bboxes,
                                    std::vector<float>& out);
    static void extraFeatures     (const std::vector<cv::Mat>& frames,
                                    const std::vector<cv::Rect>& bboxes,
                                    int img_area,
                                    std::vector<float>& out);

    static std::vector<double> dftMagnitude(const std::vector<double>& signal);
};