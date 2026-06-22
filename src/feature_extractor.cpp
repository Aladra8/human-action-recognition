// File: feature_extractor.hpp
// Author: Gulce Sirvanci (Student ID: 2087153)
// Project: Human Action Recognition 


#include "feature_extractor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

//  Internal stat helpers 
namespace {

template<typename T>
double vmean(const std::vector<T>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double vstd(const std::vector<T>& v) {
    if (v.size() < 2) return 0.0;
    double m = vmean(v), s = 0.0;
    for (auto x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

template<typename T>
T vmax(const std::vector<T>& v) {
    return v.empty() ? T{} : *std::max_element(v.begin(), v.end());
}

float sdiv(float a, float b, float fb = 0.f) {
    return std::abs(b) < 1e-6f ? fb : a / b;
}

} // namespace

//  DFT magnitude spectrum 
std::vector<double> FeatureExtractor::dftMagnitude(const std::vector<double>& sig) {
    int n = static_cast<int>(sig.size());
    std::vector<double> mag(n / 2 + 1, 0.0);
    for (int k = 0; k <= n / 2; ++k) {
        double re = 0, im = 0;
        for (int t = 0; t < n; ++t) {
            double a = 2.0 * M_PI * k * t / n;
            re += sig[t] * std::cos(a);
            im -= sig[t] * std::sin(a);
        }
        mag[k] = std::sqrt(re * re + im * im) / n;
    }
    return mag;
}

//  f0-f7 : centroid velocity 
void FeatureExtractor::velocityFeatures(const std::vector<cv::Rect>& bboxes,
                                         int img_w, std::vector<float>& out) {
    std::vector<float> cx, cy;
    for (const auto& b : bboxes)
        if (!b.empty() && b.width > 0) {
            cx.push_back(b.x + b.width  * 0.5f);
            cy.push_back(b.y + b.height * 0.5f);
        }

    if (cx.size() < 2) { out.insert(out.end(), 8, 0.f); return; }

    float norm = std::max(static_cast<float>(img_w), 1.f);
    std::vector<float> vx, vy, spd;
    for (int i = 1; i < (int)cx.size(); ++i) {
        float dx = cx[i] - cx[i-1], dy = cy[i] - cy[i-1];
        vx.push_back(dx); vy.push_back(dy);
        spd.push_back(std::sqrt(dx*dx + dy*dy));
    }

    float vxm  = static_cast<float>(vmean(vx))  / norm;
    float vym  = static_cast<float>(vmean(vy))  / norm;
    float spdm = static_cast<float>(vmean(spd)) / norm;

    out.push_back(std::abs(vxm));                           // f0 mean horizontal speed
    out.push_back(static_cast<float>(vstd(vx))  / norm);   // f1 horizontal speed std
    out.push_back(spdm);                                    // f2 mean overall speed
    out.push_back(static_cast<float>(vmax(spd)) / norm);   // f3 peak speed
    out.push_back(static_cast<float>(vstd(spd)) / norm);   // f4 speed variability
    out.push_back(std::abs(cx.back()-cx.front()) / norm);  // f5 total horizontal displacement
    out.push_back(std::abs(cy.back()-cy.front()) / norm);  // f6 total vertical displacement
    out.push_back(sdiv(std::abs(vxm), std::abs(vym)+1e-4f)); // f7 horizontal/vertical ratio
}

//  f8-f19 : dense optical flow 
void FeatureExtractor::flowFeatures(const std::vector<cv::Mat>& frames,
                                     const std::vector<cv::Rect>& bboxes,
                                     std::vector<float>& out) {
    int n = static_cast<int>(std::min(frames.size(), bboxes.size())) - 1;
    if (n <= 0) { out.insert(out.end(), 12, 0.f); return; }

    std::vector<float> all_mag, all_fx, all_fy;
    std::vector<float> upper_fx, upper_fy, lower_fx, lower_fy;
    std::vector<float> left_ufx, right_ufx;  // for clapping convergence signal

    for (int i = 0; i < n; ++i) {
        cv::Mat g1, g2;
        if (frames[i].channels()   == 3) cv::cvtColor(frames[i],   g1, cv::COLOR_BGR2GRAY);
        else g1 = frames[i];
        if (frames[i+1].channels() == 3) cv::cvtColor(frames[i+1], g2, cv::COLOR_BGR2GRAY);
        else g2 = frames[i+1];

        cv::Rect bbox = bboxes[i];
        if (bbox.empty() || bbox.width < 8 || bbox.height < 8) continue;
        cv::Rect cl = bbox & cv::Rect(0, 0, g1.cols, g1.rows);
        if (cl.width < 8 || cl.height < 8) continue;

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(g1(cl), g2(cl), flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        int rows = flow.rows, cols = flow.cols;
        int hh = rows / 2, hw = cols / 2;

        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                cv::Vec2f fv = flow.at<cv::Vec2f>(r, c);
                float fx = fv[0], fy = fv[1];
                all_mag.push_back(std::sqrt(fx*fx + fy*fy));
                all_fx.push_back(fx); all_fy.push_back(fy);
                if (r < hh) {
                    upper_fx.push_back(fx); upper_fy.push_back(fy);
                    (c < hw ? left_ufx : right_ufx).push_back(fx);
                } else { lower_fx.push_back(fx); lower_fy.push_back(fy); }
            }
    }

    auto sm = [](const std::vector<float>& v) -> float {
        return v.empty() ? 0.f : static_cast<float>(vmean(v));
    };

    float ufx = sm(upper_fx), ufy = sm(upper_fy);
    float lfx = sm(lower_fx), lfy = sm(lower_fy);
    float umag = std::sqrt(ufx*ufx + ufy*ufy);
    float lmag = std::sqrt(lfx*lfx + lfy*lfy);
    float fxm  = sm(all_fx),  fym = sm(all_fy);
    // Clapping: left side flows right (+), right side flows left (-)  → positive convergence
    float conv = sm(left_ufx) - sm(right_ufx);
    float hdom = sdiv(std::abs(fxm), std::abs(fxm) + std::abs(fym) + 1e-6f);

    out.push_back(sm(all_mag));                         // f8  mean flow magnitude
    out.push_back(static_cast<float>(vstd(all_mag)));  // f9  flow variability
    out.push_back(std::abs(fxm));                      // f10 mean horizontal flow
    out.push_back(std::abs(fym));                      // f11 mean vertical flow
    out.push_back(hdom);                               // f12 horizontal dominance
    out.push_back(ufx);                                // f13 upper body fx (signed)
    out.push_back(ufy);                                // f14 upper body fy (signed)
    out.push_back(lfx);                                // f15 lower body fx (signed)
    out.push_back(lfy);                                // f16 lower body fy (signed)
    out.push_back(sdiv(umag, lmag + 1e-4f));           // f17 upper/lower motion ratio
    out.push_back(conv);                               // f18 clapping convergence signal
    out.push_back(static_cast<float>(vstd(all_fy)));  // f19 vertical flow std
}

//  f20-f24 : bounding-box shape 
void FeatureExtractor::shapeFeatures(const std::vector<cv::Rect>& bboxes,
                                      int img_w, int img_h,
                                      std::vector<float>& out) {
    std::vector<float> areas, aspects, heights;
    float ia = static_cast<float>(img_w * img_h);
    for (const auto& b : bboxes) {
        if (b.empty() || b.width <= 0) continue;
        float w = b.width, h = b.height;
        areas.push_back(w * h);
        aspects.push_back(h / (w + 1e-4f));
        heights.push_back(h);
    }
    out.push_back(sdiv(static_cast<float>(vmean(areas)),   ia));      // f20 norm area
    out.push_back(sdiv(static_cast<float>(vstd(areas)),    ia));      // f21 area variability
    out.push_back(static_cast<float>(vmean(aspects)));                // f22 mean h/w ratio
    out.push_back(static_cast<float>(vstd(aspects)));                 // f23 ratio variability
    out.push_back(sdiv(static_cast<float>(vmean(heights)),            // f24 norm height
                       static_cast<float>(img_h)));
}

//  f25-f28 : stride frequency via FFT 
void FeatureExtractor::frequencyFeatures(const std::vector<cv::Rect>& bboxes,
                                          std::vector<float>& out) {
    std::vector<double> h_sig, cy_sig;
    for (const auto& b : bboxes)
        if (!b.empty() && b.width > 0) {
            h_sig.push_back(b.height);
            cy_sig.push_back(b.y + b.height * 0.5);
        }

    if (h_sig.size() < 8) { out.insert(out.end(), 4, 0.f); return; }

    // Remove mean (DC) before FFT.
    auto detrend = [](std::vector<double>& s) {
        double m = vmean(s);
        for (auto& x : s) x -= m;
    };
    detrend(h_sig); detrend(cy_sig);

    auto fft_h  = dftMagnitude(h_sig);
    auto fft_cy = dftMagnitude(cy_sig);

    // Find the dominant frequency bin (skip DC at k=0).
    auto dominant = [](const std::vector<double>& spec, int nf) -> std::pair<float,float> {
        double mx = 0; int ki = 1;
        for (int k = 1; k < (int)spec.size(); ++k)
            if (spec[k] > mx) { mx = spec[k]; ki = k; }
        return {static_cast<float>(ki) / nf, static_cast<float>(mx)};
    };

    int nf = static_cast<int>(h_sig.size());
    auto [hf, hp] = dominant(fft_h,  nf);
    auto [cf, cp] = dominant(fft_cy, nf);

    out.push_back(hf);   // f25 dominant stride freq from bbox height
    out.push_back(hp);   // f26 power at that frequency
    out.push_back(cf);   // f27 dominant freq from centroid-y
    out.push_back(cp);   // f28 power at that frequency
}

//  f29-f33 : discriminative extras 
void FeatureExtractor::extraFeatures(const std::vector<cv::Mat>& frames,
                                      const std::vector<cv::Rect>& bboxes,
                                      int img_area,
                                      std::vector<float>& out) {
    int n = static_cast<int>(std::min(frames.size(), bboxes.size())) - 1;
    if (n <= 0) { out.insert(out.end(), 5, 0.f); return; }

    // Per-frame upper-body flow data for temporal variance
    std::vector<float> upper_mag_per_frame;
    std::vector<float> left_ufx_total, right_ufx_total;
    std::vector<float> upper_fy_total;  // positive=down, negative=up

    // Speed per frame (for acceleration)
    std::vector<float> speeds;
    {
        std::vector<float> cx, cy;
        for (const auto& b : bboxes)
            if (!b.empty() && b.width > 0) {
                cx.push_back(b.x + b.width  * 0.5f);
                cy.push_back(b.y + b.height * 0.5f);
            }
        for (int i = 1; i < (int)cx.size(); ++i) {
            float dx = cx[i]-cx[i-1], dy = cy[i]-cy[i-1];
            speeds.push_back(std::sqrt(dx*dx + dy*dy));
        }
    }

    // Max foreground coverage across all frames
    float max_fg_frac = 0.f;

    for (int i = 0; i < n; ++i) {
        cv::Mat g1, g2;
        if (frames[i].channels()   == 3) cv::cvtColor(frames[i],   g1, cv::COLOR_BGR2GRAY);
        else g1 = frames[i];
        if (frames[i+1].channels() == 3) cv::cvtColor(frames[i+1], g2, cv::COLOR_BGR2GRAY);
        else g2 = frames[i+1];

        cv::Rect bbox = bboxes[i];
        if (bbox.empty() || bbox.width < 8 || bbox.height < 8) continue;
        cv::Rect cl = bbox & cv::Rect(0, 0, g1.cols, g1.rows);
        if (cl.width < 8 || cl.height < 8) continue;

        // Max foreground coverage using simple abs-diff within bbox region
        cv::Mat bg_local;
        cv::GaussianBlur(g1(cl), bg_local, {5,5}, 0);
        cv::Mat diff_local;
        cv::absdiff(g1(cl), g2(cl), diff_local);
        float fg_frac = static_cast<float>(cv::countNonZero(diff_local > 20))
                        / static_cast<float>(cl.area() + 1);
        max_fg_frac = std::max(max_fg_frac, fg_frac);

        // Dense optical flow in bbox
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(g1(cl), g2(cl), flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        int rows = flow.rows, cols = flow.cols;
        int hh = rows / 2, hw = cols / 2;

        float frame_upper_mag = 0.f;
        int   upper_count = 0;

        for (int r = 0; r < hh; ++r) {
            for (int c = 0; c < cols; ++c) {
                cv::Vec2f fv = flow.at<cv::Vec2f>(r, c);
                float fx = fv[0], fy = fv[1];
                float mag = std::sqrt(fx*fx + fy*fy);
                frame_upper_mag += mag;
                upper_count++;
                upper_fy_total.push_back(fy);
                (c < hw ? left_ufx_total : right_ufx_total).push_back(fx);
            }
        }
        if (upper_count > 0)
            upper_mag_per_frame.push_back(frame_upper_mag / upper_count);
    }

    auto smean = [](const std::vector<float>& v) -> float {
        return v.empty() ? 0.f : static_cast<float>(vmean(v));
    };

    // f29: left-right flow asymmetry in upper body
    float lmean = smean(left_ufx_total);
    float rmean = smean(right_ufx_total);
    float asym  = std::abs(lmean - rmean) /
                  (std::abs(lmean) + std::abs(rmean) + 1e-4f);

    // f30: net upward flow in upper body (negative fy = upward in image)
    //      normalised to [-1, 1]. Positive = upward dominant (waving).
    float upward = -smean(upper_fy_total);  // flip sign: upward motion = positive

    // f31: temporal std of upper-body flow magnitude (variability)
    float mag_var = static_cast<float>(vstd(upper_mag_per_frame));

    // f32: speed acceleration std (change in frame-to-frame speed)
    float accel_std = 0.f;
    if (speeds.size() >= 3) {
        std::vector<float> deltas;
        for (int i = 1; i < (int)speeds.size(); ++i)
            deltas.push_back(std::abs(speeds[i] - speeds[i-1]));
        accel_std = static_cast<float>(vstd(deltas));
    }

    // f33: max foreground coverage
    out.push_back(asym);                                          // f29
    out.push_back(upward / (std::abs(upward) + 1e-4f) *          // f30 (normalised)
                  std::min(1.f, std::abs(upward) * 5.f));
    out.push_back(mag_var);                                       // f31
    out.push_back(accel_std / (static_cast<float>(
                               frames.empty() ? 1 : frames[0].cols) + 1e-4f)); // f32
    out.push_back(max_fg_frac);                                   // f33
}

// Public entry point 
FeatureVector FeatureExtractor::extract(const std::vector<cv::Mat>& frames,
                                         const std::vector<cv::Rect>& bboxes) {
    FeatureVector fv;
    fv.features.reserve(FEATURE_DIM);

    int img_w = frames.empty() ? 320 : frames[0].cols;
    int img_h = frames.empty() ? 240 : frames[0].rows;

    velocityFeatures  (bboxes, img_w,                      fv.features); // f0 –f7
    flowFeatures      (frames, bboxes,                     fv.features); // f8 –f19
    shapeFeatures     (bboxes, img_w, img_h,               fv.features); // f20–f24
    frequencyFeatures (bboxes,                              fv.features); // f25–f28
    extraFeatures     (frames, bboxes, img_w * img_h,      fv.features); // f29–f33

    return fv;
}
