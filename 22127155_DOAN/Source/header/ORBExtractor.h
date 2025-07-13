#ifndef ORB_EXTRACTOR_H
#define ORB_EXTRACTOR_H

#include "LocalFeature.h"
#include <string>  // Add this for string type
#include <opencv2/features2d.hpp>

class ORBExtractor : public LocalFeature {
public:
    ORBExtractor(int nFeatures = 1000);
    std::vector<float> extract(const cv::Mat& image) override;
    std::string getMethodName() const override;  
    double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) override;
    size_t getFeatureDimension() const override { return 32 * nFeatures; } // 32 là chiều descriptor ORB mặc định
protected:
    cv::Mat computeDescriptors(const cv::Mat& image) override;
private:
    int nFeatures;  
};

#endif