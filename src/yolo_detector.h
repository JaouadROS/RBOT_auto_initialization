#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "onnx/autobackend.h"

class YoloDetector {
public:
    YoloDetector(const std::string& modelPath, const std::string& logid, const std::string& provider, float confThreshold, float iouThreshold, float maskThreshold, int colorConversionCode)
        : model(modelPath.c_str(), logid.c_str(), provider.c_str()),
          confThreshold(confThreshold),
          iouThreshold(iouThreshold),
          maskThreshold(maskThreshold),
          colorConversionCode(colorConversionCode) {
        names = model.getNames();
        skeleton = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};
    }

    std::map<std::string, cv::Mat> detectAndPlot(cv::Mat& frame,
                                                cv::Mat& imageAnnotated,
                                                std::vector<YoloResults>& detectedObjects);

    std::map<std::string, cv::Mat> plot_results(cv::Mat& img,
                                                        std::vector<YoloResults>& results,
                                                        std::unordered_map<int, std::string>& names);

private:
    AutoBackendOnnx model;
    float confThreshold;
    float iouThreshold;
    float maskThreshold;
    int colorConversionCode;
    std::unordered_map<int, std::string> names;
    std::vector<std::vector<int>> skeleton;
};