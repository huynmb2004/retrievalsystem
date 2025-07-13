#include "ColorCorrelogram.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

ColorCorrelogram::ColorCorrelogram(int bins, const std::vector<int>& dists, bool hsv)
    : colorBins(bins), distances(dists), useHSV(hsv) {}

std::vector<float> ColorCorrelogram::extract(const cv::Mat& image) {
    cv::Mat smallImg;
    cv::resize(image, smallImg, cv::Size(32, 32)); // Downscale for speed
    // Bước 1: Lượng tử hóa ảnh
    cv::Mat quantized = quantizeImage(image);
    
    // Bước 2: Tính toán correlogram
    cv::Mat correlogram;
    computeAutoCorrelogram(quantized, correlogram);
    
    // Chuẩn hóa correlogram
    cv::normalize(correlogram, correlogram, 1.0, 0.0, cv::NORM_L1);
    
    // Chuyển đổi sang vector<float>
    std::vector<float> features;
    if (correlogram.isContinuous()) {
        features.assign((float*)correlogram.datastart, (float*)correlogram.dataend);
    } else {
        for (int i = 0; i < correlogram.rows; ++i) {
            features.insert(features.end(), correlogram.ptr<float>(i), 
                          correlogram.ptr<float>(i) + correlogram.cols);
        }
    }
    
    return features;
}

double ColorCorrelogram::compare(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    // Sử dụng khoảng cách Euclidean
    double sum = 0.0;
    for (size_t i = 0; i < feat1.size(); ++i) {
        double diff = feat1[i] - feat2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

cv::Mat ColorCorrelogram::quantizeImage(const cv::Mat& image) {
    cv::Mat processedImage;
    
    if (useHSV) {
        // Chuyển sang không gian màu HSV
        cv::cvtColor(image, processedImage, cv::COLOR_BGR2HSV);
        
        // Chia kênh
        cv::Mat channels[3];
        cv::split(processedImage, channels);
        
        // Lượng tử hóa từng kênh
        cv::Mat hue, sat, val;
        cv::normalize(channels[0], hue, 0, colorBins-1, cv::NORM_MINMAX, CV_8U);
        cv::normalize(channels[1], sat, 0, 3, cv::NORM_MINMAX, CV_8U);
        cv::normalize(channels[2], val, 0, 3, cv::NORM_MINMAX, CV_8U);
        
        // Kết hợp thành ảnh lượng tử hóa
        processedImage = hue + sat * colorBins + val * colorBins * 4;
        // Make sure processedImage is CV_32S
        processedImage.convertTo(processedImage, CV_32S);
    } else {
        // Lượng tử hóa trực tiếp ảnh RGB
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        
        for (auto& channel : channels) {
            cv::normalize(channel, channel, 0, colorBins-1, cv::NORM_MINMAX, CV_8U);
        }
        
        cv::merge(channels, processedImage);
        
        // Chuyển đổi sang single channel với mã màu kết hợp
        cv::Mat combined(image.rows, image.cols, CV_32S);
        
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                cv::Vec3b pixel = processedImage.at<cv::Vec3b>(i, j);
                combined.at<int>(i, j) = pixel[0] + pixel[1] * colorBins + pixel[2] * colorBins * colorBins;
            }
        }
        
        processedImage = combined; // Keep as CV_32S
    }
    
    return processedImage;
}

void ColorCorrelogram::computeAutoCorrelogram(const cv::Mat& quantized, cv::Mat& correlogram) {
    int numColors = useHSV ? colorBins * 4 * 4 : colorBins * colorBins * colorBins;
    correlogram = cv::Mat::zeros(numColors, distances.size(), CV_32F);
    for (size_t d = 0; d < distances.size(); ++d) {
        computeCorrelogramForDistance(quantized, correlogram, d); // truyền chỉ số
    }
}

void ColorCorrelogram::computeCorrelogramForDistance(const cv::Mat& quantized, cv::Mat& correlogram, int distIdx) {
    int distance = distances[distIdx];
    // Các hướng để kiểm tra (4 hướng cơ bản)
    const int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    
    // Ma trận đếm tạm thời
    cv::Mat count = cv::Mat::zeros(correlogram.rows, 1, CV_32F);
    
    // Duyệt qua từng pixel ảnh
    for (int y = 0; y < quantized.rows; ++y) {
        for (int x = 0; x < quantized.cols; ++x) {
            int currentColor = quantized.at<int>(y, x);
            
            // Kiểm tra các pixel lân cận ở khoảng cách distance
            for (int dir = 0; dir < 4; ++dir) {
                int nx = x + directions[dir][0] * distance;
                int ny = y + directions[dir][1] * distance;
                
                if (nx >= 0 && nx < quantized.cols && ny >= 0 && ny < quantized.rows) {
                    int neighborColor = quantized.at<int>(ny, nx);
                    if (currentColor == neighborColor) {
                        count.at<float>(currentColor, 0) += 1.0f;
                    }
                }
            }
        }
    }
    
    // Chuẩn hóa và lưu vào correlogram
    float totalPairs = quantized.rows * quantized.cols * 4.0f; // 4 hướng
    for (int c = 0; c < correlogram.rows; ++c) {
        correlogram.at<float>(c, distIdx) = count.at<float>(c, 0) / totalPairs;
    }
}