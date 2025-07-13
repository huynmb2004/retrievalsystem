#ifndef COLOR_CORRELOGRAM_H
#define COLOR_CORRELOGRAM_H

#include "FeatureExtractor.h"
#include <vector>

class ColorCorrelogram : public FeatureExtractor {
private:
    int colorBins;
    std::vector<int> distances;
    bool useHSV;
    
public:
    ColorCorrelogram(int bins = 8, const std::vector<int>& dists = {1, 3, 5}, bool hsv = true);
    
    std::vector<float> extract(const cv::Mat& image) override;
    double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) override;
    std::string getMethodName() const override { return "ColorCorrelogram"; }
    size_t getFeatureDimension() const override { return colorBins*3; }
private:
    cv::Mat quantizeImage(const cv::Mat& image);
    void computeAutoCorrelogram(const cv::Mat& quantized, cv::Mat& correlogram);
    void computeCorrelogramForDistance(const cv::Mat& quantized, cv::Mat& correlogram, int distance);
};

#endif