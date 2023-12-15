#include "yolo_detector.h" // Include your YOLO wrapper header

std::map<std::string, cv::Mat> YoloDetector::detectAndPlot(cv::Mat& frame, cv::Mat& imageAnnotated, std::vector<YoloResults>& detectedObjects) {
    frame.copyTo(imageAnnotated);
    detectedObjects = model.predict_once(imageAnnotated, confThreshold, iouThreshold, maskThreshold, colorConversionCode);
    cv::cvtColor(imageAnnotated, imageAnnotated, cv::COLOR_RGB2BGR);
    return plot_results(imageAnnotated, detectedObjects, names);
}

std::map<std::string, cv::Mat> YoloDetector::plot_results(cv::Mat& img,
                                                        std::vector<YoloResults>& results,
                                                        std::unordered_map<int, std::string>& names) {
    int radius = 5;
    bool drawLines = true;

    auto raw_image_shape = img.size();

    std::map<std::string, cv::Mat> masks;
    for (const auto& res : results) {
        float left = res.bbox.x;
        float top = res.bbox.y;
        int color_num = res.class_idx;

        // Draw bounding box
        rectangle(img, res.bbox, cv::Scalar(0, 255, 0), 2);

        // Try to get the class name corresponding to the given class_idx
        std::string class_name;
        auto it = names.find(res.class_idx);
        if (it != names.end()) {
            class_name = it->second;
        }
        else {
            std::cerr << "Warning: class_idx not found in names for class_idx = " << res.class_idx << std::endl;
            // Then convert it to a string anyway
            class_name = std::to_string(res.class_idx);
        }

        // Draw mask if available
        cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
        if (res.mask.rows && res.mask.cols > 0) {
            mask(res.bbox).setTo(cv::Scalar(255, 255, 255), res.mask);
        }

        std::string identifier = names[res.class_idx];
        masks[identifier] = mask;

        // Create label
        /*std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2) << res.conf;
        std::string label = labelStream.str();

        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rect_to_fill, cv::Scalar(0, 255, 0), -1);
        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);

        // Check if keypoints are available
        if (!res.keypoints.empty()) {
            auto keypoint = res.keypoints;
            bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
            drawLines &= isPose;

            // draw points
            for (int i = 0; i < 17; i++) {
                int idx = i * 3;
                int x_coord = static_cast<int>(keypoint[idx]);
                int y_coord = static_cast<int>(keypoint[idx + 1]);

                if (x_coord % raw_image_shape.width != 0 && y_coord % raw_image_shape.height != 0) {
                    if (keypoint.size() == 3) {
                        float conf = keypoint[2];
                        if (conf < 0.5) {
                            continue;
                        }
                    }
                    cv::Scalar color_k = cv::Scalar(0, 0, 255);  // Default to red if not in pose mode
                    cv::circle(img, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
                }
            }
            // draw lines
            if (drawLines) {
                for (int i = 0; i < skeleton.size(); i++) {
                    const std::vector<int> &sk = skeleton[i];
                    int idx1 = sk[0] - 1;
                    int idx2 = sk[1] - 1;

                    int idx1_x_pos = idx1 * 3;
                    int idx2_x_pos = idx2 * 3;

                    int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                    int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                    int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                    int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                    float conf1 = keypoint[idx1_x_pos + 2];
                    float conf2 = keypoint[idx2_x_pos + 2];

                    // Check confidence thresholds
                    if (conf1 < 0.5 || conf2 < 0.5) {
                        continue;
                    }

                    // Check if positions are within bounds
                    if (x1 % raw_image_shape.width == 0 || y1 % raw_image_shape.height == 0 || x1 < 0 || y1 < 0 ||
                        x2 % raw_image_shape.width == 0 || y2 % raw_image_shape.height == 0 || x2 < 0 || y2 < 0) {
                        continue;
                    }

                    // Draw a line between keypoints
                    cv::Scalar color_limb = cv::Scalar(255, 0, 0);
                    cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
                }
            }
        }*/
    }

    // Combine the image and mask
    //addWeighted(img, 0.6, mask, 0.4, 0, img);

    return masks;
}