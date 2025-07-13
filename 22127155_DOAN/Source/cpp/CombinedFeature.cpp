#include "CombinedFeature.h"
#include <stdexcept>

using namespace cv;
using namespace std;

CombinedFeature::CombinedFeature(vector<unique_ptr<FeatureExtractor>>&& extractors, 
                               const vector<double>& weights)
    : extractors(move(extractors)), weights(weights) {
    if (this->extractors.empty()) {
        throw invalid_argument("At least one extractor must be provided");
    }
    if (this->extractors.size() != this->weights.size()) {
        throw invalid_argument("Number of extractors must match number of weights");
    }
    // Lưu số chiều đặc trưng
    for (const auto& ext : this->extractors) {
        featureDims.push_back(ext->getFeatureDimension());
    }
}

vector<float> CombinedFeature::extract(const Mat& image) {
    if (image.empty()) {
        throw runtime_error("Empty image provided to CombinedFeature");
    }

    vector<float> combinedFeatures;
    for (auto& extractor : extractors) {
        Mat processedImage = image.clone();
        if (processedImage.channels() > 1 && 
            (extractor->getMethodName().find("Color") == string::npos)) {
            // Convert to grayscale for non-color features
            cvtColor(processedImage, processedImage, COLOR_BGR2GRAY);
        }

        if (processedImage.empty()) {
            throw runtime_error("Empty Processedimage provided to CombinedFeature");
        }
        
        auto features = extractor->extract(processedImage);
        combinedFeatures.insert(combinedFeatures.end(), features.begin(), features.end());
    }
    return combinedFeatures;
}

double CombinedFeature::compare(const vector<float>& feat1, const vector<float>& feat2) {
    double totalDistance = 0.0;
    size_t startIdx = 0;
    for (size_t i = 0; i < extractors.size(); ++i) {
        size_t featSize = featureDims[i];
        if (startIdx + featSize > feat1.size() || startIdx + featSize > feat2.size()) {
            throw std::runtime_error("CombinedFeature::compare: Feature vector size mismatch or extractor returned fewer features than expected.");
        }
        vector<float> subFeat1(feat1.begin() + startIdx, feat1.begin() + startIdx + featSize);
        vector<float> subFeat2(feat2.begin() + startIdx, feat2.begin() + startIdx + featSize);
        totalDistance += weights[i] * extractors[i]->compare(subFeat1, subFeat2);
        startIdx += featSize;
    }
    return totalDistance;
}

string CombinedFeature::getMethodName() const {
    string name = "Combined_";
    for (const auto& extractor : extractors) {
        name += extractor->getMethodName() + "+";
    }
    name.pop_back(); // Remove last '+'
    return name;
}