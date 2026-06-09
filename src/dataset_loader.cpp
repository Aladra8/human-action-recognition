#include "dataset_loader.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

namespace fs = std::filesystem;

// Known class directory names in the KTH dataset.
static const std::set<std::string> KTH_CLASSES = {
    "boxing", "handclapping", "handwaving", "jogging", "running", "walking"
};

static int classDirToId(const std::string& name) {
    if (name == "walking")      return 1;
    if (name == "jogging")      return 2;
    if (name == "running")      return 3;
    if (name == "boxing")       return 4;
    if (name == "handwaving")   return 5;
    if (name == "handclapping") return 6;
    return -1;
}

//Constructor 
DatasetLoader::DatasetLoader(const std::string& path) : dataset_path_(path) {}

// Static helpers 
bool DatasetLoader::isNormalised(const GroundTruth& gt) {
    return (gt.x_center <= 1.5f && gt.y_center <= 1.5f &&
            gt.width    <= 1.5f && gt.height   <= 1.5f);
}

cv::Rect DatasetLoader::toAbsoluteBBox(const GroundTruth& gt, int img_w, int img_h) {
    float xc = gt.x_center, yc = gt.y_center, w = gt.width, h = gt.height;
    if (isNormalised(gt)) {
        xc *= img_w; yc *= img_h;
        w  *= img_w; h  *= img_h;
    }
    return cv::Rect(static_cast<int>(xc - w / 2.f),
                    static_cast<int>(yc - h / 2.f),
                    static_cast<int>(w),
                    static_cast<int>(h));
}

std::vector<std::string> DatasetLoader::sortedImages(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

//  Parse one gt.txt 
bool DatasetLoader::parseGroundTruth(const std::string& txt, GroundTruth& gt) {
    std::ifstream f(txt);
    if (!f.is_open()) return false;
    f >> gt.class_id >> gt.x_center >> gt.y_center >> gt.width >> gt.height;
    return !f.fail() && gt.class_id >= 1 && gt.class_id <= 6;
}

//  Load one sequence folder 
bool DatasetLoader::loadSequence(const std::string& folder, Sequence& seq) {
    auto images = sortedImages(folder);
    if (images.empty()) return false;

    // Find gt.txt
    for (const auto& e : fs::directory_iterator(folder)) {
        if (e.is_regular_file() && e.path().extension() == ".txt") {
            if (!parseGroundTruth(e.path().string(), seq.gt))
                std::cerr << "[WARN] Bad GT in " << folder << "\n";
            break;
        }
    }

    int count = std::min(static_cast<int>(images.size()), 40);
    seq.frames.reserve(count);
    for (int i = 0; i < count; ++i) {
        cv::Mat img = cv::imread(images[i]);
        if (img.empty()) { std::cerr << "[WARN] Cannot read " << images[i] << "\n"; continue; }
        seq.frames.push_back(img);
    }
    if (seq.frames.empty()) return false;

    seq.folder_path = folder;
    seq.name        = fs::path(folder).filename().string();

    if (seq.gt.class_id > 0)
        seq.gt_bbox_abs = toAbsoluteBBox(seq.gt,
                                          seq.frames[0].cols,
                                          seq.frames[0].rows);
    return true;
}

// Load all sequences from root 
std::vector<Sequence> DatasetLoader::loadAll() {
    std::vector<Sequence> seqs;
    fs::path root(dataset_path_);

    if (!fs::exists(root)) {
        std::cerr << "[ERROR] Dataset path not found: " << dataset_path_ << "\n";
        return seqs;
    }

    // Detect whether root contains class-named sub-directories.
    bool has_class_dirs = false;
    for (const auto& e : fs::directory_iterator(root)) {
        if (e.is_directory() && KTH_CLASSES.count(e.path().filename().string())) {
            has_class_dirs = true; break;
        }
    }

    if (has_class_dirs) {
        // Layout A: root/class_name/seq_folder/
        for (const auto& cls_dir : fs::directory_iterator(root)) {
            if (!cls_dir.is_directory()) continue;
            int cls_id = classDirToId(cls_dir.path().filename().string());
            for (const auto& seq_dir : fs::directory_iterator(cls_dir.path())) {
                if (!seq_dir.is_directory()) continue;
                Sequence seq;
                if (loadSequence(seq_dir.path().string(), seq)) {
                    if (seq.gt.class_id < 1 && cls_id > 0) seq.gt.class_id = cls_id;
                    seqs.push_back(std::move(seq));
                }
            }
        }
    } else {
        // Layout B: root/seq_folder/ (class from gt.txt)
        for (const auto& e : fs::directory_iterator(root)) {
            if (!e.is_directory()) continue;
            Sequence seq;
            if (loadSequence(e.path().string(), seq))
                seqs.push_back(std::move(seq));
        }
    }

    std::sort(seqs.begin(), seqs.end(),
              [](const Sequence& a, const Sequence& b){ return a.name < b.name; });

    std::cout << "[INFO] Loaded " << seqs.size() << " sequences from "
              << dataset_path_ << "\n";
    return seqs;
}
