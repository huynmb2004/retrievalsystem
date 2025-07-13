#include "FeatureExtractor.h"
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

// Extract features from the image
string FeatureExtractor::featuresToString(const vector<float>& features) {
    ostringstream oss;
    for (size_t i = 0; i < features.size(); ++i) {
        if (i != 0) oss << ",";
        oss << features[i];
    }
    return oss.str();
}

// Convert a comma-separated string to a vector of features
vector<float> FeatureExtractor::stringToFeatures(const string& featureStr) {
    vector<float> features;
    stringstream ss(featureStr);
    string item;
    
    while (getline(ss, item, ',')) {
        features.push_back(stof(item));
    }
    
    return features;
}