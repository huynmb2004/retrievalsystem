#include "LocalFeature.h"
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

// Constructor for LocalFeature, initializes the feature detector.
vector<float> LocalFeature::extract(const Mat& image) {
    Mat descriptors = computeDescriptors(image);
    
    // Convert descriptors matrix to vector<float>
    vector<float> features;
    if (descriptors.isContinuous()) {
        features.assign((float*)descriptors.datastart, (float*)descriptors.dataend);
    } else {
        for (int i = 0; i < descriptors.rows; ++i) {
            features.insert(features.end(), descriptors.ptr<float>(i), 
                          descriptors.ptr<float>(i) + descriptors.cols);
        }
    }
    return features;
}

// Computes the descriptors for the given image using the initialized feature detector.
Mat LocalFeature::computeDescriptors(const Mat& image) {
    if (!detector) {
        throw runtime_error("Feature detector not initialized");
    }

    vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(image, noArray(), keypoints, descriptors);
    
    if (descriptors.empty()) {
        return Mat::zeros(1, 128, CV_32F);
    }
    
    return descriptors;
}