// File: action_classifier.hpp
// Author: Gulce Sirvanci (Student ID: 2087153)
// Project: Human Action Recognition 

#include "action_classifier.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

// SVM factory 
cv::Ptr<cv::ml::SVM> ActionClassifier::makeSVM() {
    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setC(50.0);
    svm->setGamma(0.05);
    svm->setTermCriteria({cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 2000, 1e-6});
    return svm;
}

ActionClassifier::ActionClassifier() : svm_(makeSVM()) {}

// Feature matrix helper 
cv::Mat ActionClassifier::toMat(const std::vector<FeatureVector>& fvs) {
    if (fvs.empty()) return {};
    int rows = static_cast<int>(fvs.size());
    int cols = static_cast<int>(fvs[0].features.size());
    cv::Mat m(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m.at<float>(i, j) = fvs[i].features[j];
    return m;
}

// Min-max normalisation 
std::vector<FeatureVector> ActionClassifier::normalise(const std::vector<FeatureVector>& in,
                                                        std::vector<float>& mn,
                                                        std::vector<float>& mx) {
    if (in.empty()) return {};
    int dim = static_cast<int>(in[0].features.size());
    mn.assign(dim, std::numeric_limits<float>::max());
    mx.assign(dim, std::numeric_limits<float>::lowest());

    for (const auto& fv : in)
        for (int j = 0; j < dim; ++j) {
            mn[j] = std::min(mn[j], fv.features[j]);
            mx[j] = std::max(mx[j], fv.features[j]);
        }

    std::vector<FeatureVector> out(in.size());
    for (int i = 0; i < (int)in.size(); ++i) {
        out[i].features.resize(dim);
        for (int j = 0; j < dim; ++j) {
            float rng = mx[j] - mn[j];
            out[i].features[j] = rng < 1e-7f ? 0.f : (in[i].features[j] - mn[j]) / rng;
        }
    }
    return out;
}

FeatureVector ActionClassifier::normaliseOne(const FeatureVector& fv,
                                              const std::vector<float>& mn,
                                              const std::vector<float>& mx) {
    FeatureVector r;
    int dim = static_cast<int>(fv.features.size());
    r.features.resize(dim);
    for (int j = 0; j < dim; ++j) {
        float rng = mx[j] - mn[j];
        r.features[j] = rng < 1e-7f ? 0.f : (fv.features[j] - mn[j]) / rng;
    }
    return r;
}

// Train 
void ActionClassifier::train(const std::vector<FeatureVector>& features,
                              const std::vector<int>& labels) {
    if (features.empty()) throw std::runtime_error("Empty training set");

    auto normed = normalise(features, mn_, mx_);
    cv::Mat feat = toMat(normed);
    cv::Mat lbl(static_cast<int>(labels.size()), 1, CV_32S);
    for (int i = 0; i < (int)labels.size(); ++i) lbl.at<int>(i) = labels[i];

    svm_ = makeSVM();
    svm_->train(feat, cv::ml::ROW_SAMPLE, lbl);
    trained_ = true;
    std::cout << "[INFO] SVM trained on " << features.size() << " samples.\n";
}

// ── Predict ───────────────────────────────────────────────────────────────────
int ActionClassifier::predict(const FeatureVector& fv) const {
    if (!trained_) throw std::runtime_error("Model not trained");
    auto n = normaliseOne(fv, mn_, mx_);
    return static_cast<int>(svm_->predict(toMat({n})));
}

// ── Persist ───────────────────────────────────────────────────────────────────
void ActionClassifier::save(const std::string& path) const {
    if (!trained_) return;
    svm_->save(path);
    cv::FileStorage fs(path + ".norm", cv::FileStorage::WRITE);
    fs << "min" << mn_ << "max" << mx_;
}

void ActionClassifier::load(const std::string& path) {
    svm_ = cv::ml::SVM::load(path);
    cv::FileStorage fs(path + ".norm", cv::FileStorage::READ);
    fs["min"] >> mn_;
    fs["max"] >> mx_;
    trained_ = true;
}

// ── Leave-One-Out Cross Validation ───────────────────────────────────────────
ActionClassifier::LOOCVResult
ActionClassifier::runLOOCV(const std::vector<FeatureVector>& features,
                            const std::vector<int>& labels) {
    int n = static_cast<int>(features.size());
    LOOCVResult res;
    res.predictions.resize(n, -1);
    res.confusion_matrix.assign(7, std::vector<int>(7, 0));

    int correct = 0;
    for (int test = 0; test < n; ++test) {
        // Build train split (all except this one sample).
        std::vector<FeatureVector> tr_f;
        std::vector<int>           tr_l;
        for (int i = 0; i < n; ++i) {
            if (i == test) continue;
            tr_f.push_back(features[i]);
            tr_l.push_back(labels[i]);
        }

        std::vector<float> mn, mx;
        auto normed_tr   = normalise(tr_f, mn, mx);
        auto normed_test = normaliseOne(features[test], mn, mx);

        auto svm = makeSVM();
        cv::Mat feat = toMat(normed_tr);
        cv::Mat lbl(static_cast<int>(tr_l.size()), 1, CV_32S);
        for (int i = 0; i < (int)tr_l.size(); ++i) lbl.at<int>(i) = tr_l[i];
        svm->train(feat, cv::ml::ROW_SAMPLE, lbl);

        int pred = static_cast<int>(svm->predict(toMat({normed_test})));
        res.predictions[test] = pred;

        int gt = labels[test];
        if (pred == gt) ++correct;
        if (gt >= 1 && gt <= 6 && pred >= 1 && pred <= 6)
            res.confusion_matrix[gt][pred]++;

        // Progress every 12 sequences (one full class).
        if ((test + 1) % 12 == 0 || test == n - 1)
            std::cout << "  LOOCV " << (test+1) << "/" << n
                      << "  acc so far = "
                      << 100.0 * correct / (test + 1) << "%\n";
    }
    res.accuracy = static_cast<double>(correct) / n;
    return res;
}
