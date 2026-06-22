// File: feature_extractor.hpp
// Author: Baba Drammeh (Student ID: 2085440)
// Project: Human Action Recognition 

#pragma once
#include "types.hpp"
#include <string>
#include <vector>

// EvalResult – all metrics collected in one place.
struct EvalResult {
    double mIoU            = 0.0;   // mean IoU over all 72 sequences (frame 20)
    double global_accuracy = 0.0;   // fraction of sequences correctly classified
    std::vector<double> per_seq_iou;
    std::vector<double> f1;         // index 1–6 valid; index 0 unused
    // confusion_matrix[true_class][pred_class] – 7×7, index 0 unused
    std::vector<std::vector<int>> confusion_matrix;
    std::vector<int> predictions;
    std::vector<int> ground_truths;
};

// Evaluator – computes and reports performance metrics.
class Evaluator {
public:
    // Intersection over Union for two axis-aligned rectangles.
    static double IoU(const cv::Rect& pred, const cv::Rect& gt);

    // Assemble a full EvalResult from raw predictions + IoUs.
    static EvalResult compute(const std::vector<int>& predictions,
                               const std::vector<int>& ground_truths,
                               const std::vector<double>& ious);

    // Print a formatted report to stdout.
    static void printReport(const EvalResult& r);

    // Save report as CSV for use in Excel / spreadsheet.
    static void saveCSV(const EvalResult& r, const std::string& path);

private:
    static std::vector<double>           f1Scores      (const std::vector<int>& preds,
                                                         const std::vector<int>& gts);
    static std::vector<std::vector<int>> confusionMatrix(const std::vector<int>& preds,
                                                          const std::vector<int>& gts);
};
