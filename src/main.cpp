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
#include "yolo_detector.h"

using namespace std;
using namespace cv;

//#define DEBUG_MODE
//#define ALIGN_DEBUG_MODE

void drawResultOverlay(const vector<Object3D*>& objects, const cv::Mat& frame, cv::Mat& result)
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
}

void drawResultOverlay(const Object3D* object, const cv::Mat& frame, cv::Mat& result)
{
    // render the models with phong shading
    RenderingEngine::Instance()->setLevel(0);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(1.0, 0.5, 0.0));
    std::vector<Model*> objectVector;
    objectVector.push_back(const_cast<Model*>(reinterpret_cast<const Model*>(object)));
    RenderingEngine::Instance()->renderShaded(objectVector, GL_FILL, colors, true);

    
    // download the rendering to the CPU
    Mat rendering = RenderingEngine::Instance()->downloadFrame(RenderingEngine::RGB);
    
    // download the depth buffer to the CPU
    Mat depth = RenderingEngine::Instance()->downloadFrame(RenderingEngine::DEPTH);
    
    // compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
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

void alignTxTy(float& tx, float& ty, const Point2f& centroid_proj, const Point2f& centroid_det, float& step_size, bool& tx_ty_aligned) {
    float distance = norm(centroid_proj - centroid_det);

    // Dynamically adjust the step size based on the distance
    float dynamic_step_size = std::max(static_cast<float>(3 * step_size * (distance / 10.0)), 0.01f);

    if (centroid_proj.x < centroid_det.x) tx += dynamic_step_size;
    else if (centroid_proj.x > centroid_det.x) tx -= dynamic_step_size;

    if (centroid_proj.y < centroid_det.y) ty += dynamic_step_size;
    else if (centroid_proj.y > centroid_det.y) ty -= dynamic_step_size;

    // Update the alignment threshold as needed
    const float alignmentThreshold = 1.0;
    if (distance < alignmentThreshold) {
        tx_ty_aligned = true;
        //cout << "Alignment achieved for tx and ty." << endl;
    }
}

void alignTz(float& tz, const Rect& boundingBox_proj, const Rect& boundingBox_det, float& tz_step_size, bool& tz_aligned) {
    const float tz_alignment_threshold_x = 10.0;
    const float tz_alignment_threshold_y = 10.0;
    const float tz_area_alignment_threshold = 0.05; // Threshold for alignment

    float x_difference = abs(boundingBox_proj.x - boundingBox_det.x);
    float y_difference = abs(boundingBox_proj.y - boundingBox_det.y);
    float area_proj = boundingBox_proj.area();
    float area_det = boundingBox_det.area();
    float area_difference = abs(area_proj - area_det) / area_det;

    // Adjust the step size based on the area difference
    float dynamic_tz_step_size = std::max(4 * tz_step_size * sqrt(area_difference), 0.01f); // Adjust scaling factor and minimum step size as needed

    if ((area_difference < tz_area_alignment_threshold)) {// || (x_difference < tz_alignment_threshold_x || y_difference < tz_alignment_threshold_y)) {
        tz_aligned = true;
        //cout << "Alignment achieved for tz with value: " << tz << endl;
    }
    else {
        if (area_proj < area_det) tz -= dynamic_tz_step_size;
        else if (area_proj > area_det) tz += dynamic_tz_step_size;
    }
}

void alignObject(PoseEstimator6D* poseEstimator, Object3D* object, Mat& frame, const Mat& mask, float& tx, float& ty, float& tz, float& alpha, float& beta, float& gamma) {
    std::vector<cv::Point> contour_mask_gray;
    Mat mask_gray;
    std::vector<cv::Point> contour_projected_model;
    // Find contours in the grayscale image
    cvtColor(mask, mask_gray, cv::COLOR_BGR2GRAY);
    std::vector<std::vector<cv::Point>> contours_mask_gray;
    findContours(mask_gray, contours_mask_gray, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    contour_mask_gray = findLargestContour(contours_mask_gray);

    // Initializations and constants
    float step_size = 0.1; // For tx and ty
    float tz_step_size = 1; // For tz

    bool tx_ty_aligned = false;
    bool tz_aligned = false;

    // Generate the transformation matrix
    Matx44f T_cm = Transformations::translationMatrix(tx, ty, tz)
                    * Transformations::rotationMatrix(alpha, Vec3f(1, 0, 0))
                    * Transformations::rotationMatrix(beta, Vec3f(0, 1, 0))
                    * Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1))
                    * Matx44f::eye();
    object->setInitialPose(T_cm);

    std::future<void> txTyTask, tzTask;

    while (true) {
        poseEstimator->estimatePoses(frame, false, true);
        Mat projected_object = Mat::zeros(frame.size(), frame.type());
        drawResultOverlay(object, projected_object, projected_object);

        // Find contours in the grayscale image
        Mat gray;
        cvtColor(projected_object, gray, cv::COLOR_BGR2GRAY);
        std::vector<std::vector<cv::Point>> contours_projected_model;
        findContours(gray, contours_projected_model, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        contour_projected_model = findLargestContour(contours_projected_model);

        #ifdef ALIGN_DEBUG_MODE 
        cv::drawContours(projected_object, contours_mask_gray, -1, cv::Scalar(0, 0, 255), 2);
        cv::drawContours(projected_object, contours_projected_model, -1, cv::Scalar(0, 255, 0), 2);
        #endif

        // Draw the contours and calculate centroids
        Point2f centroid_proj, centroid_det;
        if (!contours_projected_model.empty()) {
            centroid_proj = calculateCentroid(contour_projected_model);
            #ifdef ALIGN_DEBUG_MODE 
            circle(projected_object, centroid_proj, 5, Scalar(0, 255, 0), -1); // Centroid as green circle
            #endif
        }
        if (!contours_mask_gray.empty()) {
            centroid_det = calculateCentroid(contour_mask_gray);
            #ifdef ALIGN_DEBUG_MODE 
            circle(projected_object, centroid_det, 5, Scalar(0, 0, 255), -1); // Centroid as red circle
            #endif
        }

        // Draw the contours and calculate centroids
        Rect boundingBox_proj, boundingBox_det;
        if (!contours_projected_model.empty()) {
            boundingBox_proj = boundingRect(contour_projected_model);
            #ifdef ALIGN_DEBUG_MODE 
            rectangle(projected_object, boundingBox_proj, Scalar(0, 255, 0), 2); // Projected bounding box in green
            #endif
        }
        if (!contours_mask_gray.empty()) {
            boundingBox_det = boundingRect(contour_mask_gray);
            #ifdef ALIGN_DEBUG_MODE
            rectangle(projected_object, boundingBox_det, Scalar(0, 0, 255), 2); // Detected bounding box in red
            #endif
        }

        if (!tx_ty_aligned) {
            txTyTask = std::async(std::launch::async, [&] {
                alignTxTy(std::ref(tx), std::ref(ty), 
                            centroid_proj, centroid_det, 
                            step_size, std::ref(tx_ty_aligned));
            });
        }

        if (!tz_aligned) {
            tzTask = std::async(std::launch::async, [&] {
                alignTz(std::ref(tz), 
                            boundingBox_proj, boundingBox_det, 
                            tz_step_size, std::ref(tz_aligned));
            });
        }

        if (!tx_ty_aligned) {
            txTyTask.wait(); // Wait for txTyTask to complete if it was started
        }

        if (!tz_aligned) {
            tzTask.wait(); // Wait for tzTask to complete if it was started
        }

        // Generate the transformation matrix
        T_cm = Transformations::translationMatrix(tx, ty, tz)
                        * Transformations::rotationMatrix(alpha, Vec3f(1, 0, 0))
                        * Transformations::rotationMatrix(beta, Vec3f(0, 1, 0))
                        * Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1))
                        * Matx44f::eye();

        object->setPose(T_cm);
        object->setInitialPose(T_cm);


        if (tx_ty_aligned && tz_aligned)
            break;

        #ifdef ALIGN_DEBUG_MODE 
        cv::imshow("projected_object", projected_object);
        cv::waitKey(1);
        #endif
    }
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
    
    //Start rbot initialization
    float tx = 0.0, ty = 0.0, tz = 600;
    float alpha = 0.0, beta = 0.0, gamma = 180;

    auto start = std::chrono::high_resolution_clock::now();

    // load 3D objects
    vector<Object3D*> objects;
    objects.push_back(new Object3D("data/a.obj", tx, ty, tz, alpha, beta, gamma, 1, 0.5f, distances));
    objects.push_back(new Object3D("data/b.obj", tx, ty, tz, alpha, beta, gamma, 1, 0.5f, distances));
    objects.push_back(new Object3D("data/c.obj", tx, ty, tz, alpha, beta, gamma, 1, 0.5f, distances));
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Objects pushback done: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // create the pose estimator
    PoseEstimator6D* poseEstimator = new PoseEstimator6D(width, height, zNear, zFar, K, distCoeffs, objects);
    
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "poseEstimator initialization done: " << duration.count() << " seconds" << std::endl;
    
    // move the OpenGL context for offscreen rendering to the current thread, if run in a seperate QT worker thread (unnessary in this example)
    //RenderingEngine::Instance()->getContext()->moveToThread(this);
    
    // active the OpenGL context for the offscreen rendering engine during pose estimation
    RenderingEngine::Instance()->makeCurrent();

    start = std::chrono::high_resolution_clock::now();
    //YoloDetector yoloDetector("config/three_objects.onnx", "yolov8_inference2", OnnxProviders::CUDA, 0.30f, 0.45f, 0.50f, cv::COLOR_BGR2RGB);
    YoloDetector yoloDetector("config/best640640.onnx", "yolov8_inference2", OnnxProviders::CUDA, 0.30f, 0.45f, 0.50f, cv::COLOR_BGR2RGB);
    string folderPath = "data/three_objects_animation/";

    int count = 120; //Select the image frame number
    string fileName = folderPath + format("%04d.jpg", count);
    Mat frame = imread(fileName);//TODO change that later to be the first frame

    std::cout<<"Start Yolo detection"<<std::endl;
    //Yolo detection
    cv::Mat imageAnnotated;
    std::vector<YoloResults> detectedObjects;
    std::map<std::string, cv::Mat> objectMasks = yoloDetector.detectAndPlot(frame, imageAnnotated, detectedObjects);

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Yolo detection done: " << duration.count() << " seconds" << std::endl;

    cv::Mat result;
    imageAnnotated.copyTo(result);
    drawResultOverlay(objects, frame, result);
    cv::imshow("result", result);
    cv::waitKey(1);

    start = std::chrono::high_resolution_clock::now();
    int objectsIndex = 0;
    for (const auto& objectMask : objectMasks) {
        const cv::Mat& mask = objectMask.second;

        float tx = 0.0, ty = 0.0, tz = 600;
        float alpha = 0.0, beta = 0.0, gamma = 180;
        alignObject(poseEstimator, objects[objectsIndex], frame, mask, tx, ty, tz, alpha, beta, gamma);
        #ifdef DEBUG_MODE
            imageAnnotated.copyTo(result);
            drawResultOverlay(objects, frame, result);
            cv::imshow("result", result);
            cv::waitKey(1);
        #endif
        objectsIndex++;
    }

    poseEstimator->estimatePoses(frame, false, true);

    cv::imshow("result", result);
    cv::waitKey(1);

    for (int i = 0; i < objectMasks.size(); i++) {
        poseEstimator->toggleTracking(frame, i, false);
        poseEstimator->estimatePoses(frame, false, false);
    }

    //Optimize the pose in 100 iterations maximum
    for(int i = 0; i < 100; i++)
    {
        int energy = 0;
        for (int index = 0; index < objectMasks.size(); index++) {
            float energyFunctionValue = poseEstimator->getEnergyFunctionValue(index);
            if(energyFunctionValue <= 0.23){
                energy++;
            }
        }

        //Break when all the objects can be tracked <=> energyFunctionValue <= 0.23
        if(energy == objectMasks.size()){
            break;
        }

        poseEstimator->estimatePosesForInitialization(frame);

        #ifdef DEBUG_MODE
            imageAnnotated.copyTo(result);
            drawResultOverlay(objects, frame, result);
            cv::imshow("result", result);
            cv::waitKey(1);
        #endif
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Initialization tracking done: " << duration.count() << " seconds" << std::endl;

    while(true)
    {
        // obtain an input image
        string fileName = folderPath + format("%04d.jpg", count);
        frame = imread(fileName);
        count++;
        if (frame.empty()) {

            cerr << "Error reading image: " << endl;
            break;  // Skip to the next image if reading fails
        }
        
        // the main pose uodate call
        poseEstimator->estimatePoses(frame, false, true);
        
        // render the models with the resulting pose estimates ontop of the input image
        frame.copyTo(result);
        drawResultOverlay(objects, frame, result);

        imshow("result", result);
        
        int key = waitKey(1);
        
        //cout << objects[0]->getPose() << endl;
        
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
