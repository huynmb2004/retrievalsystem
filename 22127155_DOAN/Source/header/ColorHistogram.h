#ifndef COLOR_HISTOGRAM_H
#define COLOR_HISTOGRAM_H

#include "FeatureExtractor.h"

class ColorHistogram : public FeatureExtractor {
private:
    int binsPerChannel;
    bool useHSV;
    
public:
    ColorHistogram(int bins = 8, bool hsv = true);
    
    std::vector<float> extract(const cv::Mat& image) override;
    double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) override;
    std::string getMethodName() const override { return "ColorHistogram"; }
    size_t getFeatureDimension() const override { return binsPerChannel * 3; }
private:
    cv::Mat computeHistogram(const cv::Mat& image);
    void normalizeHistogram(cv::Mat& hist);
};

#endif