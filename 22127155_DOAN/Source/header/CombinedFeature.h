#ifndef COMBINED_FEATURE_H
#define COMBINED_FEATURE_H

#include "FeatureExtractor.h"
#include <vector>
#include <memory>

class CombinedFeature : public FeatureExtractor {
public:
    CombinedFeature(std::vector<std::unique_ptr<FeatureExtractor>>&& extractors, 
                   const std::vector<double>& weights);
    
    std::vector<float> extract(const cv::Mat& image) override;
    double compare(const std::vector<float>& feat1, const std::vector<float>& feat2) override;
    std::string getMethodName() const override;
    size_t getFeatureDimension() const override {
        size_t totalDim = 0;
        for (const auto& dim : featureDims) {
            totalDim += dim;
        }
        return totalDim;
    }
    
private:
    std::vector<std::unique_ptr<FeatureExtractor>> extractors;
    std::vector<double> weights;
    std::vector<size_t> featureDims; // Thêm dòng này
};

#endif