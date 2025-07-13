#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;
    
    // Phương thức trừu tượng để trích xuất đặc trưng
    virtual std::vector<float> extract(const cv::Mat& image) = 0;
    
    // Phương thức tính toán khoảng cách giữa 2 đặc trưng
    virtual double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) = 0;
    
    // Lấy tên phương pháp trích xuất (dùng cho header CSV)
    virtual std::string getMethodName() const = 0;
    
    // Chuyển đặc trưng thành string để lưu vào CSV
    virtual std::string featuresToString(const std::vector<float>& features);
    
    // Chuyển string từ CSV thành vector đặc trưng
    virtual std::vector<float> stringToFeatures(const std::string& featureStr);

    // Thêm dòng này:
    virtual size_t getFeatureDimension() const = 0;
};

#endif