#pragma once
#include "types.hpp"
#include <string>
#include <vector>

// DatasetLoader
//
// Scans a root folder and returns all valid 40-frame sequences.
//
// Supported layouts:
//   A) root/class_name/sequence_folder/   (class inferred from directory name)
//   B) root/sequence_folder/              (class read from gt.txt)
//
// Each sequence folder must contain:
//   • Up to 40 image files (.jpg / .jpeg / .png / .bmp), sorted alphabetically
//   • One .txt file with: <class_id> <x_c> <y_c> <w> <h>
class DatasetLoader {
public:
    explicit DatasetLoader(const std::string& dataset_path);

    std::vector<Sequence> loadAll();

    // Convert a GroundTruth bbox to absolute pixels given image dimensions.
    static cv::Rect toAbsoluteBBox(const GroundTruth& gt, int img_w, int img_h);

private:
    std::string dataset_path_;

    bool loadSequence(const std::string& folder, Sequence& seq);
    bool parseGroundTruth(const std::string& txt_file, GroundTruth& gt);

    static std::vector<std::string> sortedImages(const std::string& folder);
    static bool isNormalised(const GroundTruth& gt);
};
