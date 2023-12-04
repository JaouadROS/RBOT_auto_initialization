/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#include <QApplication>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "object3d.h"
#include "pose_estimator6d.h"

//For yolo/Onnx
#include "onnx/autobackend.h"

using namespace std;
using namespace cv;

cv::Mat drawResultOverlay(const vector<Object3D*>& objects, const cv::Mat& frame)
{
    // render the models with phong shading
    RenderingEngine::Instance()->setLevel(0);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(1.0, 0.5, 0.0));
    //colors.push_back(Point3f(0.2, 0.3, 1.0));
    RenderingEngine::Instance()->renderShaded(vector<Model*>(objects.begin(), objects.end()), GL_FILL, colors, true);
    
    // download the rendering to the CPU
    Mat rendering = RenderingEngine::Instance()->downloadFrame(RenderingEngine::RGB);
    
    // download the depth buffer to the CPU
    Mat depth = RenderingEngine::Instance()->downloadFrame(RenderingEngine::DEPTH);
    
    // compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
    Mat result = frame.clone();
    for(int y = 0; y < frame.rows; y++)
    {
        for(int x = 0; x < frame.cols; x++)
        {
            Vec3b color = rendering.at<Vec3b>(y,x);
            if(depth.at<float>(y,x) != 0.0f)
            {
                result.at<Vec3b>(y,x)[0] = color[2];
                result.at<Vec3b>(y,x)[1] = color[1];
                result.at<Vec3b>(y,x)[2] = color[0];
            }
        }
    }
    return result;
}

// Function to calculate angle of a line formed by two points
float calculateAngle(const Point2f& a, const Point2f& b) {
    float angle = atan2(b.y - a.y, b.x - a.x) * 180.0 / CV_PI;
    return angle;
}

Point2f calculateCentroid(const vector<Point>& contour) {
    Moments m = moments(contour, false);
    return Point2f(m.m10 / m.m00, m.m01 / m.m00);
}

std::vector<cv::Point> findLargestContour(const std::vector<std::vector<cv::Point>>& contours) {
    double maxArea = 0;
    int maxAreaIdx = -1;

    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }

    // Create a vector to store the largest contour
    std::vector<cv::Point> largestContour;

    if (maxAreaIdx != -1) {
        largestContour = contours[maxAreaIdx];
    }

    return largestContour;
}

// Define the skeleton and color mappings
std::vector<std::vector<int>> skeleton = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};
Mat plot_results(cv::Mat& img, std::vector<YoloResults>& results,
                  std::unordered_map<int, std::string>& names,
                  const cv::Size& shape) {

    //cv::Mat mask = img.clone();
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());

    int radius = 5;
    bool drawLines = true;

    auto raw_image_shape = img.size();

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
        if (res.mask.rows && res.mask.cols > 0) {
            mask(res.bbox).setTo(cv::Scalar(255, 255, 255), res.mask);
        }

        // Create label
        std::stringstream labelStream;
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
        }
    }

    // Combine the image and mask
    addWeighted(img, 0.6, mask, 0.4, 0, img);

    return mask;
}

void alignObject(PoseEstimator6D* poseEstimator, vector<Object3D*>& objects, Mat& frame, Mat& mask, float& tx, float& ty, float& tz, float& alpha, float& beta, float& gamma) {
    std::vector<cv::Point> contour_mask_gray;
    Mat mask_gray;
    std::vector<cv::Point> contour_projected_model;

    // Initializations and constants
    float step_size = 0.1; // For tx and ty
    float tz_step_size = 0.1; // For tz
    float gamma_step_size = 0.5; // Adjust step size for gamma as needed
    float gamma_alignment_threshold = 8.0; // Threshold for corner distance, adjust as needed

    bool tx_ty_aligned = false;
    bool tz_aligned = false;
    bool gamma_aligned = false;

    while (true) {
        // Generate the transformation matrix
        Matx44f T_cm = Transformations::translationMatrix(tx, ty, tz)
                        * Transformations::rotationMatrix(alpha, Vec3f(1, 0, 0))
                        * Transformations::rotationMatrix(beta, Vec3f(0, 1, 0))
                        * Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1))
                        * Matx44f::eye();

        objects[0]->setPose(T_cm);

        // Find contours in the grayscale image
        Mat result;
        frame.copyTo(result);
        cvtColor(mask, mask_gray, cv::COLOR_BGR2GRAY);
        std::vector<std::vector<cv::Point>> contours_mask_gray;
        findContours(mask_gray, contours_mask_gray, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        contour_mask_gray = findLargestContour(contours_mask_gray);
        cv::drawContours(result, contours_mask_gray, -1, cv::Scalar(0, 0, 255), 2);

        // the main pose update call
        poseEstimator->estimatePoses(frame, false, true);
        Mat projected_object = Mat::zeros(frame.size(), frame.type());
        projected_object = drawResultOverlay(objects, projected_object);

        // Find contours in the grayscale image
        Mat gray;
        cvtColor(projected_object, gray, cv::COLOR_BGR2GRAY);
        std::vector<std::vector<cv::Point>> contours_projected_model;
        findContours(gray, contours_projected_model, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        contour_projected_model = findLargestContour(contours_projected_model);
        cv::drawContours(result, contours_projected_model, -1, cv::Scalar(0, 255, 0), 2);

        // Draw the contours and calculate centroids
        Point2f centroid_proj, centroid_det;
        if (!contours_projected_model.empty()) {
            centroid_proj = calculateCentroid(contour_projected_model);
            circle(result, centroid_proj, 5, Scalar(0, 255, 0), -1); // Centroid as green circle
        }
        if (!contours_mask_gray.empty()) {
            centroid_det = calculateCentroid(contour_mask_gray);
            circle(result, centroid_det, 5, Scalar(0, 0, 255), -1); // Centroid as red circle
        }

        // Draw the contours and calculate centroids
        Rect boundingBox_proj, boundingBox_det;
        if (!contours_projected_model.empty()) {
            boundingBox_proj = boundingRect(contour_projected_model);
            rectangle(result, boundingBox_proj, Scalar(0, 255, 0), 2); // Projected bounding box in green
        }
        if (!contours_mask_gray.empty()) {
            boundingBox_det = boundingRect(contour_mask_gray);
            rectangle(result, boundingBox_det, Scalar(0, 0, 255), 2); // Detected bounding box in red
        }

        if (!tx_ty_aligned) {
            // Adjust tx and ty based on the centroids
            if (centroid_proj.x < centroid_det.x) tx += step_size;
            else if (centroid_proj.x > centroid_det.x) tx -= step_size;

            if (centroid_proj.y < centroid_det.y) ty += step_size;
            else if (centroid_proj.y > centroid_det.y) ty -= step_size;

            float distance = norm(centroid_proj - centroid_det);

            const float alignmentThreshold = 1.0;
            if (distance < alignmentThreshold) {
                tx_ty_aligned = true;
                cout << "Alignment achieved for tx and ty." << endl;
            }
        }
        else if (!tz_aligned) {
            // Define a threshold for how close the coordinates need to be
            const float tz_alignment_threshold_x = 10.0; // Adjust as needed
            const float tz_alignment_threshold_y = 10.0; // Adjust as needed
            const float tz_area_alignment_threshold = 0.05; // Adjust as needed

            // Calculate the differences in the x and y coordinates of the bounding boxes
            float x_difference = abs(boundingBox_proj.x - boundingBox_det.x);
            float y_difference = abs(boundingBox_proj.y - boundingBox_det.y);

            // Calculate the relative difference in area
            float area_proj = boundingBox_proj.area();
            float area_det = boundingBox_det.area();
            float area_difference = abs(area_proj - area_det) / area_det;

            // Check if the differences are within the thresholds
            if ((area_difference < tz_area_alignment_threshold) || (x_difference < tz_alignment_threshold_x || y_difference < tz_alignment_threshold_y)) {
                tz_aligned = true;
                cout << "Alignment achieved for tz with value: " << tz << endl;
            }
            else {
                // Adjust tz based on the difference in areas
                if (area_proj < area_det) tz -= tz_step_size;
                else if (area_proj > area_det) tz += tz_step_size;
            }
        }
        else if (!gamma_aligned) {
            // Calculate diagonal angles for both bounding boxes
            float angle_proj = calculateAngle(boundingBox_proj.tl(), boundingBox_proj.br());
            float angle_det = calculateAngle(boundingBox_det.tl(), boundingBox_det.br());

            // Determine if the projected object needs to rotate clockwise or counterclockwise
            float angle_difference = angle_proj - angle_det;
            std::cout<<angle_difference<<std::endl;

            if (abs(angle_difference) < gamma_alignment_threshold) {
                gamma_aligned = true;
                cout << "Alignment achieved for gamma with value: " << gamma << endl;
                tx_ty_aligned = false; // go back to tx ty correction due to gamma rotation
            }
            else {
                // Adjust gamma based on the angle difference
                if (angle_difference > 0) gamma -= gamma_step_size;
                else gamma += gamma_step_size;
            }
        }

        if (tx_ty_aligned && tz_aligned && gamma_aligned)
            break;

        cv::imshow("result", result);
        cv::waitKey(1);
    }
}

void adjustAlphaBeta(PoseEstimator6D* poseEstimator, vector<Object3D*>& objects, Mat& frame, Mat& mask, float& tx, float& ty, float& tz, float& alpha, float& beta, float& gamma) {
    const int alpha_beta_range = 20; // Range of values for alpha and beta (-20 to +20)
    const float step_size = 1;     // Step size for adjusting alpha and beta
    
    cv::Mat result;
    for (int alpha_offset = -alpha_beta_range; alpha_offset <= alpha_beta_range; ++alpha_offset) {
        for (int beta_offset = -alpha_beta_range; beta_offset <= alpha_beta_range; ++beta_offset) {
            // Apply the offset values to alpha and beta
            float new_alpha = alpha_offset * step_size;
            float new_beta = beta_offset * step_size;

            // Generate the transformation matrix
            Matx44f T_cm = Transformations::translationMatrix(tx, ty, tz)
                        * Transformations::rotationMatrix(new_alpha, Vec3f(1, 0, 0))
                        * Transformations::rotationMatrix(new_beta, Vec3f(0, 1, 0))
                        * Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1))
                        * Matx44f::eye();

            objects[0]->setPose(T_cm);

            // Call toggleTracking to check initialization
            result = drawResultOverlay(objects, frame);
            poseEstimator->toggleTracking(frame, 0, false);

            cout << "on going: " << objects[0]->isTrackingLost() << " " << new_alpha << " and beta: " << new_beta << endl;
            cv::imshow("adjustAlphaBeta", result);
            cv::waitKey(1);
            if (!objects[0]->isTrackingLost()) {
                // Initialization is successful
                cout << "Initialization successful for alpha: " << new_alpha << " and beta: " << new_beta << endl;
                beta = new_beta;
                alpha = new_alpha;

                return;
            }
        }
    }

    // If initialization was not successful for any combination of alpha and beta
    cout << "Initialization for alpha and beta not successful within the specified range." << endl;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // camera image size
    int width = 1920;
    int height = 1080;
    
    // near and far plane of the OpenGL view frustum
    float zNear = 10.0;
    float zFar = 10000.0;
    
    // camera instrinsics
    Matx33f K = Matx33f(1.5999999e+03, 0, 960, 0, 1.5999999e+03, 540, 0, 0, 1);
    Matx14f distCoeffs =  Matx14f(0.0, 0.0, 0.0, 0.0);
    
    // distances for the pose detection template generation
    vector<float> distances = {200.0f, 400.0f, 600.0f};
    
    // load 3D objects
    vector<Object3D*> objects;
    objects.push_back(new Object3D("data/squirrel.obj", 0, 1, 600, 0, 0, 180, 1, 0.5f, distances));
    //objects.push_back(new Object3D("data/a_second_model.obj", -50, 0, 600, 30, 0, 180, 1.0, 0.55f, distances2));
    
    // create the pose estimator
    PoseEstimator6D* poseEstimator = new PoseEstimator6D(width, height, zNear, zFar, K, distCoeffs, objects);
    
    // move the OpenGL context for offscreen rendering to the current thread, if run in a seperate QT worker thread (unnessary in this example)
    //RenderingEngine::Instance()->getContext()->moveToThread(this);
    
    // active the OpenGL context for the offscreen rendering engine during pose estimation
    RenderingEngine::Instance()->makeCurrent();
    
    bool showHelp = true;
    
    Mat frame = imread("data/frame.jpg");//TODO change that later to be the first frame
    //Yolo initialization
    const std::string modelPath = "config/best.onnx";
    const std::string onnx_logid_("yolov8_inference2");
    const std::string& onnx_provider_(OnnxProviders::CUDA);
    AutoBackendOnnx model(modelPath.c_str(), onnx_logid_.c_str(), onnx_provider_.c_str());
    std::unordered_map<int, std::string> names;
    names = model.getNames();
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;
    float mask_threshold = 0.50f;
    int conversion_code = cv::COLOR_BGR2RGB;

    //Yolo detection
    cv::Mat image_annotated;
    frame.copyTo(image_annotated);
    std::vector<YoloResults> objs = model.predict_once(image_annotated, conf_threshold, iou_threshold, mask_threshold, conversion_code);
    cv::Size show_shape = image_annotated.size();
    cv::cvtColor(image_annotated, image_annotated, cv::COLOR_RGB2BGR);
    Mat mask = plot_results(image_annotated, objs, names, show_shape);
    cv::imshow("imageSegmentation", image_annotated);

    //Start rbot initialization
    float tx = 10.0, ty = 10.0, tz = 600;
    float alpha = 0.0, beta = 40.0, gamma = 180;

    cout << "before alignObject: " << objects[0]->isTrackingLost() << " " << alpha << " and beta: " << beta << endl;
    alignObject(poseEstimator, objects, frame, mask, tx, ty, tz, alpha, beta, gamma);
    cout << "after alignObject: " << objects[0]->isTrackingLost() << " " << alpha << " and beta: " << beta << endl;
    adjustAlphaBeta(poseEstimator, objects, frame, mask, tx, ty, tz, alpha, beta, gamma);
    cout << "after adjustAlphaBeta: " << objects[0]->isTrackingLost() << " " << alpha << " and beta: " << beta << endl;

    //Apply the initialization
    Matx44f T_cm = Transformations::translationMatrix(tx, ty, tz) // tz = 0 for now, adjust as needed
                    * Transformations::rotationMatrix(alpha, Vec3f(1, 0, 0)) // alpha = 0 for now, adjust as needed
                    * Transformations::rotationMatrix(beta, Vec3f(0, 1, 0)) // beta = 0 for now, adjust as needed
                    * Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1)) // gamma = 0 for now, adjust as needed
                    * Matx44f::eye();

    objects[0]->setPose(T_cm);

    int timeout = 0;
    while(true)
    {
        // obtain an input image
        frame = imread("data/frame.jpg");
        
        // the main pose uodate call
        poseEstimator->estimatePoses(frame, false, true);
        
        //cout << objects[0]->getPose() << endl;
        
        // render the models with the resulting pose estimates ontop of the input image
        Mat result = drawResultOverlay(objects, frame);
        
        if(showHelp)
        {
            putText(result, "Press '1' to initialize", Point(150, 250), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 1);
            putText(result, "or 'c' to quit", Point(205, 285), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 1);
        }
        
        imshow("result", result);
        
        int key = waitKey(timeout);
        
        cout << "Tracking: " << objects[0]->isTrackingLost() << " " << alpha << " and beta: " << beta << endl;
        // start/stop tracking the first object
        if(key == (int)'1')
        {
            poseEstimator->toggleTracking(frame, 0, false);
            poseEstimator->estimatePoses(frame, false, false);
            timeout = 1;
            showHelp = !showHelp;
        }
        if(key == (int)'2') // the same for a second object
        {
            //poseEstimator->toggleTracking(frame, 1, false);
            //poseEstimator->estimatePoses(frame, false, false);
        }
        // reset the system to the initial state
        if(key == (int)'r')
            poseEstimator->reset();
        // stop the demo
        if(key == (int)'c')
            break;
    }
    
    // deactivate the offscreen rendering OpenGL context
    RenderingEngine::Instance()->doneCurrent();
    
    // clean up
    RenderingEngine::Instance()->destroy();
    
    for(int i = 0; i < objects.size(); i++)
    {
        delete objects[i];
    }
    objects.clear();
    
    delete poseEstimator;
}
