#ifndef TEXTURE_FEATURE_H
#define TEXTURE_FEATURE_H

#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>

class TextureFeature : public FeatureExtractor {
public:
    TextureFeature();
    
    std::vector<float> extract(const cv::Mat& image) override;
    double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) override;
    std::string getMethodName() const override;
    size_t getFeatureDimension() const override { return 8; } 
private:
    cv::Mat computeLBP(const cv::Mat& image);
};

#endif