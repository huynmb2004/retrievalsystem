#include "ColorHistogram.h"
#include <opencv2/imgproc.hpp>

ColorHistogram::ColorHistogram(int bins, bool hsv) 
    : binsPerChannel(bins), useHSV(hsv) {}

std::vector<float> ColorHistogram::extract(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "[ColorHistogram] Input image is empty!" << std::endl;
        return {};
    }
    cv::Mat processedImage;
    
    // Chuyển đổi màu nếu cần
    if (useHSV) {
        cv::cvtColor(image, processedImage, cv::COLOR_BGR2HSV);
    } else {
        image.copyTo(processedImage);
    }
    
    // Tính toán histogram
    cv::Mat hist = computeHistogram(processedImage);
    
    // Chuẩn hóa histogram
    normalizeHistogram(hist);
    
    // Chuyển đổi sang vector<float>
    std::vector<float> features;
    features.assign((float*)hist.datastart, (float*)hist.dataend);
    
    return features;
}

double ColorHistogram::compare(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    // Sử dụng khoảng cách Euclidean
    double sum = 0.0;
    for (size_t i = 0; i < feat1.size(); ++i) {
        double diff = feat1[i] - feat2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

cv::Mat ColorHistogram::computeHistogram(const cv::Mat& image) {
    // Thiết lập tham số histogram
    int histSize[] = {binsPerChannel, binsPerChannel, binsPerChannel};
    
    // Phạm vi giá trị màu
    float hranges[] = {0, 180}; // Hue range for HSV
    float ranges[] = {0, 256};   // Range for BGR
    const float* channelRanges[] = { 
        useHSV ? hranges : ranges, 
        ranges, 
        ranges 
    };
    
    // Kênh màu cần tính
    int channels[] = {0, 1, 2};
    
    // Tính toán histogram 3D
    cv::Mat hist;
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, channelRanges, true, false);
    
    return hist;
}

void ColorHistogram::normalizeHistogram(cv::Mat& hist) {
    // Flatten the histogram to 1D for minMaxLoc
    cv::Mat histFlat = hist.reshape(1, 1); // 1 row, all elements
    double minVal, maxVal;
    cv::minMaxLoc(histFlat, &minVal, &maxVal);

    if (maxVal > 0) {
        hist.convertTo(hist, CV_32F, 1.0 / maxVal);
    }
}