#include "ORBExtractor.h"
#include <stdexcept>  // For runtime_error
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>  // For cv::Mat and other core types

using namespace cv;
using std::vector;  // For vector type
using std::string;  // For string type
using std::runtime_error;  // For runtime_error

// Constructor for ORBExtractor, initializes the ORB detector with the specified number of features.
ORBExtractor::ORBExtractor(int nFeatures) : nFeatures(nFeatures) {
    detector = ORB::create(nFeatures);
    if (detector.empty()) {
        throw runtime_error("Failed to create ORB detector");
    }
}

// Returns the name of the method used for feature extraction.
std::vector<float> ORBExtractor::extract(const cv::Mat& image) {
    cv::Mat descriptors = computeDescriptors(image);

    // Nếu không có descriptor, trả về vector 0 đúng số chiều
    if (descriptors.empty() || descriptors.rows == 0) {
        return std::vector<float>(getFeatureDimension(), 0.0f);
    }

    // Chuyển descriptor thành vector<float>
    std::vector<float> features;
    descriptors.convertTo(descriptors, CV_32F);
    features.assign((float*)descriptors.datastart, (float*)descriptors.dataend);

    // Nếu số chiều nhỏ hơn getFeatureDimension(), bổ sung 0 cho đủ
    if (features.size() < getFeatureDimension()) {
        features.resize(getFeatureDimension(), 0.0f);
    } else if (features.size() > getFeatureDimension()) {
        features.resize(getFeatureDimension());
    }
    return features;
}

// Computes the ORB descriptors for the given image.
Mat ORBExtractor::computeDescriptors(const Mat& image) {
    if (!detector) {
        throw runtime_error("ORB detector not initialized");
    }

    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat gray;
    if (image.channels() == 3)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image;

    detector->detectAndCompute(gray, noArray(), keypoints, descriptors);

    if (descriptors.empty()) {
        return Mat::zeros(1, 32, CV_8U); // ORB descriptor size is 32 bytes
    }

    return descriptors; // Do NOT convert to CV_32F
}

double ORBExtractor::compare(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    // Kiểm tra kích thước của đặc trưng
    int dim = 32;
    if (feat1.empty() || feat2.empty()) return 9999.0;
    if (feat1.size() % dim != 0 || feat2.size() % dim != 0) return 9999.0;

    int n1 = feat1.size() / dim;
    int n2 = feat2.size() / dim;

    // Chuyển đổi vector<float> thành cv::Mat
    cv::Mat desc1(n1, dim, CV_32F, const_cast<float*>(feat1.data()));
    cv::Mat desc2(n2, dim, CV_32F, const_cast<float*>(feat2.data()));

    cv::Mat desc1_8u, desc2_8u;
    desc1.convertTo(desc1_8u, CV_8U);
    desc2.convertTo(desc2_8u, CV_8U);

    // Sử dụng BFMatcher với crossCheck = true
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // crossCheck = true
    std::vector<cv::DMatch> matches;
    matcher.match(desc1_8u, desc2_8u, matches);

    // Nếu không có matches, trả về khoảng cách lớn
    double sum = 0.0;
    for (const auto& m : matches) sum += m.distance;
    return matches.empty() ? 9999.0 : sum / (matches.size() * 32.0);
}

string ORBExtractor::getMethodName() const {
    return "ORB";
}