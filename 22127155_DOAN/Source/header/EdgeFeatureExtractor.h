#ifndef EDGE_FEATURE_EXTRACTOR_H
#define EDGE_FEATURE_EXTRACTOR_H

#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class EdgeFeatureExtractor : public FeatureExtractor {
public:
    EdgeFeatureExtractor(double threshold1 = 100, double threshold2 = 200)
        : thresh1(threshold1), thresh2(threshold2) {}

    std::vector<float> extract(const cv::Mat& image) override;
    double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) override;
    std::string getMethodName() const override { return "Edge_Canny"; }
    size_t getFeatureDimension() const override { return 8; } // 2 bins: non-edge, edge

private:
    double thresh1, thresh2;
};

#endif