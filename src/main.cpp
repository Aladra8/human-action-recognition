#include "types.hpp"
#include "dataset_loader.hpp"
#include "detector.hpp"
#include "feature_extractor.hpp"
#include "action_classifier.hpp"
#include "evaluator.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;


// processSequence
//   Background model → bbox tracking → feature extraction → IoU at frame 20
static FeatureVector processSequence(const Sequence& seq,
                                      std::vector<cv::Rect>& bboxes_out,
                                      double& iou_out) {
    bboxes_out.clear();
    iou_out = 0.0;
    if (seq.frames.empty()) return {};

    cv::Mat bg = BackgroundModel::computeMedian(seq.frames);
    bboxes_out = HumanDetector::processSequence(seq.frames, bg);

    if (static_cast<int>(bboxes_out.size()) > 19 && !seq.gt_bbox_abs.empty())
        iou_out = Evaluator::IoU(bboxes_out[19], seq.gt_bbox_abs);

    return FeatureExtractor::extract(seq.frames, bboxes_out);
}

// saveVisualOutput
//   Saves one annotated image per sequence (frame 20) into out_dir/.
//   Red box = prediction,  Green box = ground truth.
static void saveVisualOutput(const std::vector<Sequence>& seqs,
                              const std::vector<int>& preds,
                              const std::vector<std::vector<cv::Rect>>& all_bboxes,
                              const std::string& out_dir) {
    fs::create_directories(out_dir);
    int saved = 0;
    for (int s = 0; s < (int)seqs.size() && s < (int)preds.size(); ++s) {
        if (seqs[s].frames.size() <= 19 || all_bboxes[s].size() <= 19) continue;
        cv::Mat annotated = HumanDetector::drawResult(
            seqs[s].frames[19],
            all_bboxes[s][19],
            actionToString(preds[s]),
            seqs[s].gt_bbox_abs);
        cv::imwrite(out_dir + "/" + seqs[s].name + "_pred.jpg", annotated);
        ++saved;
    }
    std::cout << "[INFO] " << saved << " annotated frames saved → " << out_dir << "/\n";
}

// printUsage
static void printUsage(const char* prog) {
    std::cerr
        << "\nUsage:\n"
        << "  " << prog << " <dataset_path> [options]\n\n"
        << "Arguments:\n"
        << "  dataset_path          Folder containing the 72 sequence sub-folders\n\n"
        << "Options:\n"
        << "  --loocv               Leave-One-Out CV evaluation (default, recommended)\n"
        << "  --train-only          Train on all data; report training accuracy\n"
        << "  --visualize           Save annotated frame-20 images\n"
        << "  --output <dir>        Output folder for images  [output_frames]\n"
        << "  --report <file.csv>   CSV report path           [eval_report.csv]\n"
        << "  --save-model <file>   Save trained SVM to file\n"
        << "  --load-model <file>   Load a saved SVM (skips training)\n"
        << "  --help                Show this message\n\n"
        << "Examples:\n"
        << "  " << prog << " ./dataset --loocv --visualize\n"
        << "  " << prog << " ./dataset --loocv --save-model model.svm\n"
        << "  " << prog << " ./dataset --load-model model.svm --visualize\n\n";
}

// main
int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    std::string dataset_path, save_model, load_model;
    std::string viz_dir    = "output_frames";
    std::string report_csv = "eval_report.csv";
    bool do_loocv      = true;
    bool do_train_only = false;
    bool do_visualize  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--help")                         { printUsage(argv[0]); return 0; }
        else if (a == "--loocv")                         { do_loocv = true; do_train_only = false; }
        else if (a == "--train-only")                    { do_train_only = true; do_loocv = false; }
        else if (a == "--visualize")                     { do_visualize = true; }
        else if (a == "--save-model"  && i+1 < argc)    { save_model = argv[++i]; }
        else if (a == "--load-model"  && i+1 < argc)    { load_model = argv[++i]; }
        else if (a == "--output"      && i+1 < argc)    { viz_dir    = argv[++i]; }
        else if (a == "--report"      && i+1 < argc)    { report_csv = argv[++i]; }
        else if (a[0] != '-')                            { dataset_path = a; }
    }

    if (dataset_path.empty()) { printUsage(argv[0]); return 1; }

    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║  Human Action Recognition  –  KTH Dataset   ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n";
    std::cout << "Dataset : " << dataset_path << "\n\n";

    //  1. Load sequences 
    DatasetLoader loader(dataset_path);
    auto seqs = loader.loadAll();
    if (seqs.empty()) {
        std::cerr << "[ERROR] No sequences found. Check INSTRUCTIONS.md → Dataset section.\n";
        return 1;
    }

    // 2. Extract features from every sequence 
    std::cout << "[STEP 1/3] Feature extraction from "
              << seqs.size() << " sequences...\n";

    std::vector<FeatureVector>         all_fv;
    std::vector<int>                   all_gt;
    std::vector<double>                all_ious;
    std::vector<std::vector<cv::Rect>> all_bboxes;

    for (int s = 0; s < (int)seqs.size(); ++s) {
        std::cout << "  [" << std::setw(2) << (s+1) << "/" << seqs.size() << "] "
                  << seqs[s].name << " (class " << seqs[s].gt.class_id << ")";

        std::vector<cv::Rect> bboxes;
        double iou = 0.0;
        FeatureVector fv = processSequence(seqs[s], bboxes, iou);

        all_fv.push_back(fv);
        all_gt.push_back(seqs[s].gt.class_id);
        all_ious.push_back(iou);
        all_bboxes.push_back(bboxes);

        std::cout << "  IoU@20=" << std::fixed << std::setprecision(3) << iou << "\n";
    }

    // 3. Classify 
    std::cout << "\n[STEP 2/3] Classification...\n";
    ActionClassifier clf;
    std::vector<int> predictions;

    if (!load_model.empty()) {
        std::cout << "  Loading saved model: " << load_model << "\n";
        clf.load(load_model);
        for (const auto& fv : all_fv) predictions.push_back(clf.predict(fv));

    } else if (do_loocv) {
        std::cout << "  Leave-One-Out CV  (" << seqs.size() << " iterations)...\n";
        auto res   = ActionClassifier::runLOOCV(all_fv, all_gt);
        predictions = res.predictions;
        std::cout << "  Final LOOCV accuracy: "
                  << std::fixed << std::setprecision(2)
                  << res.accuracy * 100.0 << "%\n";
        clf.train(all_fv, all_gt);  // train final model on everything

    } else {
        clf.train(all_fv, all_gt);
        for (const auto& fv : all_fv) predictions.push_back(clf.predict(fv));
    }

    if (!save_model.empty() && clf.isTrained()) {
        clf.save(save_model);
        std::cout << "  Model saved → " << save_model << "\n";
    }

    //  4. Evaluate 
    std::cout << "\n[STEP 3/3] Evaluation...\n";
    auto eval = Evaluator::compute(predictions, all_gt, all_ious);
    Evaluator::printReport(eval);
    Evaluator::saveCSV(eval, report_csv);

    // 5. Visualise 
    if (do_visualize) {
        std::cout << "\n[VIZ] Saving annotated frame-20 images...\n";
        saveVisualOutput(seqs, predictions, all_bboxes, viz_dir);
    }

    return 0;
}
