#include "EdgeFeatureExtractor.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>

// Thay thế hàm extract trong EdgeFeatureExtractor
std::vector<float> EdgeFeatureExtractor::extract(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() > 1)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    // Làm mượt
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

    // Tính đạo hàm Sobel
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    cv::Mat magnitude, angle;
    cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);

    // Histogram hướng cạnh (8 bins)
    int bins = 8;
    std::vector<float> hist(bins, 0.0f);
    for (int y = 0; y < angle.rows; ++y) {
        for (int x = 0; x < angle.cols; ++x) {
            float mag = magnitude.at<float>(y, x);
            float ang = angle.at<float>(y, x);
            int bin = static_cast<int>(ang / 360.0 * bins) % bins;
            hist[bin] += mag;
        }
    }
    // Chuẩn hóa
    float sum = std::accumulate(hist.begin(), hist.end(), 0.0f);
    if (sum > 0) for (auto& v : hist) v /= sum;
    return hist;
}

// Compare two feature vectors
double EdgeFeatureExtractor::compare(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    // Khoảng cách Euclidean đơn giản
    if (feat1.size() != feat2.size()) return 9999.0;
    double sum = 0.0;
    for (size_t i = 0; i < feat1.size(); ++i) {
        double diff = feat1[i] - feat2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}