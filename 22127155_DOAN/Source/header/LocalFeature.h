#ifndef LOCAL_FEATURE_H
#define LOCAL_FEATURE_H

#include "FeatureExtractor.h"
#include <opencv2/features2d.hpp>

class LocalFeature : public FeatureExtractor {
protected:
    cv::Ptr<cv::Feature2D> detector;
    
public:
    std::vector<float> extract(const cv::Mat& image) override;
    
protected:
    virtual cv::Mat computeDescriptors(const cv::Mat& image);
};

#endif