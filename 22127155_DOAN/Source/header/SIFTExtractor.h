#ifndef SIFT_EXTRACTOR_H
#define SIFT_EXTRACTOR_H

#include "LocalFeature.h"
#include <opencv2/features2d.hpp>

class SIFTExtractor : public LocalFeature {
public:
    SIFTExtractor(int nFeatures = 0, int nOctaveLayers = 3, 
                 double contrastThreshold = 0.04, double edgeThreshold = 10, 
                 double sigma = 1.6);

    
    std::vector<float> extract(const cv::Mat& image) override;
    std::string getMethodName() const override;
    double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) override;
    size_t getFeatureDimension() const override { return 128 * nFeatures; } // 128 là chiều descriptor SIFT
    
protected:
    cv::Mat computeDescriptors(const cv::Mat& image) override;
    
private:
    cv::Ptr<cv::SIFT> sift;
    int nFeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
};

#endif