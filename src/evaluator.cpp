// File: evaluator.cpp
// Author: Baba Drammeh (Student ID: 2085440)
// Project: Human Action Recognition 

#include "evaluator.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

// IoU 
double Evaluator::IoU(const cv::Rect& pred, const cv::Rect& gt) {
    if (pred.empty() || gt.empty() || pred.area() == 0 || gt.area() == 0) return 0.0;
    cv::Rect inter = pred & gt;
    if (inter.empty() || inter.area() == 0) return 0.0;
    double i = inter.area();
    double u = pred.area() + gt.area() - i;
    return u > 0 ? i / u : 0.0;
}

//  Per-class F1 
std::vector<double> Evaluator::f1Scores(const std::vector<int>& preds,
                                         const std::vector<int>& gts) {
    std::vector<double> f1(7, 0.0);
    for (int c = 1; c <= 6; ++c) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < (int)preds.size(); ++i) {
            bool pc = (preds[i] == c), gc = (gts[i] == c);
            if (pc && gc) ++tp;
            else if (pc)  ++fp;
            else if (gc)  ++fn;
        }
        double prec = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        double rec  = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        f1[c] = (prec + rec > 1e-9) ? 2.0 * prec * rec / (prec + rec) : 0.0;
    }
    return f1;
}

//  Confusion matrix 
std::vector<std::vector<int>> Evaluator::confusionMatrix(const std::vector<int>& preds,
                                                          const std::vector<int>& gts) {
    std::vector<std::vector<int>> cm(7, std::vector<int>(7, 0));
    for (int i = 0; i < (int)preds.size(); ++i) {
        int g = gts[i], p = preds[i];
        if (g >= 1 && g <= 6 && p >= 1 && p <= 6) cm[g][p]++;
    }
    return cm;
}

//  Assemble 
EvalResult Evaluator::compute(const std::vector<int>& preds,
                               const std::vector<int>& gts,
                               const std::vector<double>& ious) {
    EvalResult r;
    r.predictions = preds; r.ground_truths = gts; r.per_seq_iou = ious;

    r.mIoU = ious.empty() ? 0.0 :
             std::accumulate(ious.begin(), ious.end(), 0.0) / ious.size();

    int correct = 0;
    for (int i = 0; i < (int)preds.size(); ++i) if (preds[i] == gts[i]) ++correct;
    r.global_accuracy = preds.empty() ? 0.0 : static_cast<double>(correct) / preds.size();

    r.f1              = f1Scores(preds, gts);
    r.confusion_matrix = confusionMatrix(preds, gts);
    return r;
}

//  Print 
void Evaluator::printReport(const EvalResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n--- Evaluation Report ---\n";
    std::cout << "  mIoU (frame 20)  : " << r.mIoU << "\n";
    std::cout << "  Global Accuracy  : " << r.global_accuracy * 100.0 << " %\n";

    double macro = 0;
    std::cout << "\n  Per-class F1 Scores:\n";
    for (int c = 1; c <= 6; ++c) {
        std::cout << "    " << std::setw(14) << std::left << actionToString(c)
                  << ": " << r.f1[c] << "\n";
        macro += r.f1[c];
    }
    std::cout << "    " << std::setw(14) << std::left << "Macro F1"
              << ": " << macro / 6.0 << "\n";

    std::cout << "\n  Confusion Matrix  (row = Ground Truth,  col = Predicted)\n";
    std::cout << std::setw(16) << std::left << "";
    for (int c = 1; c <= 6; ++c)
        std::cout << std::setw(7) << std::right << actionToString(c).substr(0, 6);
    std::cout << "\n";
    for (int g = 1; g <= 6; ++g) {
        std::cout << std::setw(16) << std::left << actionToString(g);
        for (int p = 1; p <= 6; ++p)
            std::cout << std::setw(7) << std::right << r.confusion_matrix[g][p];
        std::cout << "\n";
    }
    std::cout << "----------------------------------------\n";
}

// ── CSV ───────────────────────────────────────────────────────────────────────
void Evaluator::saveCSV(const EvalResult& r, const std::string& path) {
    std::ofstream f(path);
    if (!f) { std::cerr << "[WARN] Cannot write " << path << "\n"; return; }
    f << std::fixed << std::setprecision(6);
    f << "metric,value\n";
    f << "mIoU,"             << r.mIoU            << "\n";
    f << "global_accuracy,"  << r.global_accuracy  << "\n";
    for (int c = 1; c <= 6; ++c)
        f << "f1_" << actionToString(c) << "," << r.f1[c] << "\n";
    f << "\nConfusion Matrix (row=GT col=Pred)\n";
    f << ",walking,jogging,running,boxing,handwaving,handclapping\n";
    for (int g = 1; g <= 6; ++g) {
        f << actionToString(g);
        for (int p = 1; p <= 6; ++p) f << "," << r.confusion_matrix[g][p];
        f << "\n";
    }
    std::cout << "[INFO] Report saved → " << path << "\n";
}
