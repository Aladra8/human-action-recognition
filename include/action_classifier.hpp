
// File: action_classifier.hpp
// Author: Gulce Sirvanci (Student ID: 2087153)
// Project: Human Action Recognition 

#pragma once
#include "types.hpp"
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

// ActionClassifier

class ActionClassifier {
public:
    ActionClassifier();

    // Train on a labelled set of feature vectors (labels are integers 1–6).
    void train(const std::vector<FeatureVector>& features,
               const std::vector<int>& labels);

    // Predict the action class (1–6) for one feature vector.
    int predict(const FeatureVector& fv) const;

    // Save the trained model + normalisation params to disk.
    void save(const std::string& path) const;

    // Load a previously saved model from disk.
    void load(const std::string& path);

    bool isTrained() const { return trained_; }

    //  Leave-One-Out Cross Validation 
    struct LOOCVResult {
        double           accuracy = 0.0;
        std::vector<int> predictions;
        // confusion_matrix[true_class][pred_class] – 7×7, index 0 unused.
        std::vector<std::vector<int>> confusion_matrix;
    };

    // Run LOOCV on the full dataset (most honest evaluation for 72 samples).
    static LOOCVResult runLOOCV(const std::vector<FeatureVector>& features,
                                 const std::vector<int>& labels);

    //  Feature normalisation helpers 
    // Fit normaliser on training set, return normalised copy + store min/max.
    static std::vector<FeatureVector> normalise(const std::vector<FeatureVector>& in,
                                                 std::vector<float>& mn,
                                                 std::vector<float>& mx);

    // Apply fitted normaliser to a single vector.
    static FeatureVector normaliseOne(const FeatureVector& fv,
                                       const std::vector<float>& mn,
                                       const std::vector<float>& mx);

private:
    cv::Ptr<cv::ml::SVM> svm_;
    bool trained_ = false;
    std::vector<float> mn_, mx_;   // per-feature min and max

    static cv::Mat              toMat  (const std::vector<FeatureVector>& fvs);
    static cv::Ptr<cv::ml::SVM> makeSVM();
};
