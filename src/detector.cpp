// File: detector.cpp
// Author: Baba Drammeh (Student ID: 2085440)
// Project: Human Action Recognition

#include "detector.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

// BackgroundModel
cv::Mat BackgroundModel::computeMedian(const std::vector<cv::Mat>& frames) {
    if (frames.empty()) return {};

    int rows = frames[0].rows, cols = frames[0].cols;
    int n    = static_cast<int>(frames.size());

    // Convert all frames to grayscale once.
    std::vector<cv::Mat> gray(n);
    for (int i = 0; i < n; ++i) {
        if (frames[i].channels() == 3)
            cv::cvtColor(frames[i], gray[i], cv::COLOR_BGR2GRAY);
        else
            gray[i] = frames[i];
    }

    cv::Mat bg(rows, cols, CV_8UC1);
    std::vector<uchar> vals(n);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            for (int i = 0; i < n; ++i) vals[i] = gray[i].at<uchar>(r, c);
            std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
            bg.at<uchar>(r, c) = vals[n / 2];
        }

    return bg;
}

cv::Mat BackgroundModel::foregroundMask(const cv::Mat& frame,
                                         const cv::Mat& bg,
                                         int threshold) {
    cv::Mat gr;
    if (frame.channels() == 3) cv::cvtColor(frame, gr, cv::COLOR_BGR2GRAY);
    else gr = frame;

    cv::Mat diff, mask;
    cv::absdiff(gr, bg, diff);
    cv::threshold(diff, mask, threshold, 255, cv::THRESH_BINARY);
    return mask;
}

cv::Mat BackgroundModel::cleanMask(const cv::Mat& mask) {
    cv::Mat res = mask.clone();

    // Close: fills holes inside the human silhouette.
    cv::morphologyEx(res, res, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {11, 11}));

    // Open: removes small noise blobs.
    cv::morphologyEx(res, res, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

    // Dilate: ensures the bbox fully wraps the silhouette.
    cv::dilate(res, res,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

    return res;
}

// HumanDetector

cv::Rect HumanDetector::detectBBox(const cv::Mat& fg_mask, int min_area) {
    if (fg_mask.empty()) return {};

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    // Union bounding rect of all blobs large enough to be a human body part.
    cv::Rect combined;
    for (const auto& c : contours) {
        if (cv::contourArea(c) >= min_area) {
            cv::Rect r = cv::boundingRect(c);
            combined = combined.empty() ? r : (combined | r);
        }
    }
    return combined;
}

std::vector<cv::Rect> HumanDetector::processSequence(const std::vector<cv::Mat>& frames,
                                                       const cv::Mat& bg,
                                                       int bg_threshold) {
    std::vector<cv::Rect> raw;
    raw.reserve(frames.size());

    for (const auto& frame : frames) {
        cv::Mat mask = BackgroundModel::foregroundMask(frame, bg, bg_threshold);
        mask = BackgroundModel::cleanMask(mask);

        cv::Rect bbox = detectBBox(mask);
        if (!bbox.empty())
            bbox &= cv::Rect(0, 0, frame.cols, frame.rows);  // clamp
        raw.push_back(bbox);
    }
    return smoothBBoxes(raw);
}

std::vector<cv::Rect> HumanDetector::smoothBBoxes(const std::vector<cv::Rect>& raw,
                                                    float alpha) {
    if (raw.empty()) return {};

    // Seed with the first valid bbox.
    cv::Rect prev;
    for (const auto& b : raw)
        if (!b.empty() && b.width > 0) { prev = b; break; }
    if (prev.empty()) return raw;

    std::vector<cv::Rect> out = raw;
    for (auto& b : out) {
        if (b.empty() || b.width <= 0) {
            b = prev;
        } else {
            b = cv::Rect(
                static_cast<int>(alpha * b.x      + (1.f - alpha) * prev.x),
                static_cast<int>(alpha * b.y      + (1.f - alpha) * prev.y),
                std::max(1, static_cast<int>(alpha * b.width  + (1.f-alpha) * prev.width)),
                std::max(1, static_cast<int>(alpha * b.height + (1.f-alpha) * prev.height))
            );
            prev = b;
        }
    }
    return out;
}

cv::Mat HumanDetector::drawResult(const cv::Mat& frame,
                                   const cv::Rect& pred_bbox,
                                   const std::string& label,
                                   const cv::Rect& /*gt_bbox*/,    // not drawn in output
                                   cv::Scalar pred_color,
                                   cv::Scalar /*gt_color*/) {
    cv::Mat src = frame.clone();
    if (src.channels() == 1) cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);

    //  4× upscale: KTH native 160×120 → 640×480 
    constexpr int SCALE = 4;
    cv::Mat out;
    cv::resize(src, out, {src.cols * SCALE, src.rows * SCALE},
               0, 0, cv::INTER_LINEAR);

    //  Predicted bounding box (red), scaled to the new size 
    if (!pred_bbox.empty() && pred_bbox.width > 0) {
        cv::Rect scaled(pred_bbox.x      * SCALE,
                        pred_bbox.y      * SCALE,
                        pred_bbox.width  * SCALE,
                        pred_bbox.height * SCALE);
        scaled &= cv::Rect(0, 0, out.cols, out.rows);
        cv::rectangle(out, scaled, pred_color, 2);
    }

    // Label – top-right corner, matching Figure 4 in the project spec
    int    face  = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 1.1;
    int    thick = 2, base = 0;
    cv::Size ts  = cv::getTextSize(label, face, scale, thick, &base);
    cv::Point tl(out.cols - ts.width - 10, ts.height + 10);
    cv::putText(out, label, tl + cv::Point(2, 2), face, scale, {0,0,0}, thick + 2);
    cv::putText(out, label, tl,                   face, scale, pred_color, thick);

    return out;
}
