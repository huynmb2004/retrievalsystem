#include "SIFTExtractor.h"
#include <opencv2/imgproc.hpp>

// Constructor for SIFTExtractor, initializes the SIFT detector with the specified parameters.
SIFTExtractor::SIFTExtractor(int nFeatures, int nOctaveLayers, 
                           double contrastThreshold, double edgeThreshold, 
                           double sigma)
    : nFeatures(nFeatures), nOctaveLayers(nOctaveLayers),
      contrastThreshold(contrastThreshold), edgeThreshold(edgeThreshold),
      sigma(sigma) {
    sift = cv::SIFT::create(nFeatures, nOctaveLayers, 
                           contrastThreshold, edgeThreshold, sigma);
}

// Returns the name of the method used for feature extraction.
std::string SIFTExtractor::getMethodName() const {
    return "SIFT";
}

std::vector<float> SIFTExtractor::extract(const cv::Mat& image) {
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

// Extract features from the image using SIFT
double SIFTExtractor::compare(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    int dim = 128; // SIFT descriptor size
    if (feat1.empty() || feat2.empty()) {
        std::cerr << "One or both feature vectors are empty." << std::endl;
        return 0.0;
    }
    if (feat1.size() % dim != 0 || feat2.size() % dim != 0) {
        std::cerr << "Feature vector size is not a multiple of 128." << std::endl;
        return 0.0;
    }

    int n1 = feat1.size() / dim;
    int n2 = feat2.size() / dim;
    cv::Mat desc1(n1, dim, CV_32F, const_cast<float*>(feat1.data()));
    cv::Mat desc2(n2, dim, CV_32F, const_cast<float*>(feat2.data()));

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    double sum = 0.0;
    for (const auto& m : matches) sum += m.distance;
    double avg = matches.empty() ? 0.0 : sum / matches.size();

    return avg / 512.0;
}

// Convert SIFT descriptors to string for CSV storage
cv::Mat SIFTExtractor::computeDescriptors(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    // Convert to grayscale if needed
    cv::Mat grayImage;
    if (image.channels() > 1) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image;
    }
    
    sift->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors);
    
    // Convert descriptors to 1D vector (for our CBIR system)
    if (!descriptors.empty()) {
        return descriptors; // Do NOT flatten to 1 row
    }
    
    return cv::Mat();
}