#include "TextureFeature.h"
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

// Constructor for TextureFeature, initializes the feature extractor.
TextureFeature::TextureFeature() {}

// Extract texture features from the input image using Local Binary Pattern (LBP)
vector<float> TextureFeature::extract(const Mat& image) {
    if (image.empty()) {
        std::cerr << "[TextureFeature] Input image is empty!" << std::endl;
        return {};
    }
    
    Mat gray;
    if (image.channels() > 1) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    // Tính toán đặc trưng LBP (Local Binary Pattern)
    Mat lbp = computeLBP(gray);
    
    // Tính histogram LBP (8 bins)
    Mat hist;
    int histSize[] = {8};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    int channels[] = {0};
    calcHist(&lbp, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
    
    // Chuẩn hóa histogram
    normalize(hist, hist, 1.0, 0.0, NORM_L1);
    
    return vector<float>(hist.begin<float>(), hist.end<float>());
}

// Compute Local Binary Pattern (LBP) for the input image
Mat TextureFeature::computeLBP(const Mat& src) {
    // input validation
    if(src.empty()) {
        cerr << "Error: Empty image passed to computeLBP" << endl;
        return Mat();
    }
    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            uchar center = src.at<uchar>(i,j);
            unsigned char code = 0;
            code |= (src.at<uchar>(i-1,j-1) > center) << 7;
            code |= (src.at<uchar>(i-1,j) > center) << 6;
            code |= (src.at<uchar>(i-1,j+1) > center) << 5;
            code |= (src.at<uchar>(i,j+1) > center) << 4;
            code |= (src.at<uchar>(i+1,j+1) > center) << 3;
            code |= (src.at<uchar>(i+1,j) > center) << 2;
            code |= (src.at<uchar>(i+1,j-1) > center) << 1;
            code |= (src.at<uchar>(i,j-1) > center) << 0;
            dst.at<uchar>(i-1,j-1) = code;
        }
    }
    return dst;
}

// Compare two feature vectors using Euclidean distance
double TextureFeature::compare(const vector<float>& feat1, const vector<float>& feat2) {
    // Sử dụng khoảng cách Euclidean
    double sum = 0.0;
    for (size_t i = 0; i < feat1.size(); ++i) {
        double diff = feat1[i] - feat2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Get the name of the method used for feature extraction
string TextureFeature::getMethodName() const {
    return "Texture_LBP";
}