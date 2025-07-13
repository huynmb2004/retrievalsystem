#ifndef DATABASE_MANAGER_H
#define DATABASE_MANAGER_H

#include <map>
#include <string>
#include <vector>
#include <numeric>
#include "FeatureExtractor.h"

class DatabaseManager {
private:
    std::map<std::string, std::vector<float>> featuresDB;
    FeatureExtractor* extractor;
public:
    DatabaseManager(FeatureExtractor* extractor);
    ~DatabaseManager();
    
    static std::string getDatabasePath(const std::string& method, const std::string& datasetPath);
    void buildDatabase(const std::vector<std::string>& imagePaths);
    void saveDatabase(const std::string& filePath);
    bool loadDatabase(const std::string& filePath);
    
    std::vector<std::pair<std::string, double>> query(const cv::Mat& queryImage, int topK = 5);
    
    // Add these new methods
    std::string getExtractorName() const;
    // bool isEmpty() const;
    size_t getDatabaseSize() const;
    void setExtractor(FeatureExtractor* newExtractor);
    std::vector<std::pair<std::string, double>> queryWithMAP(const cv::Mat& queryImage, const std::string& queryImagePath, int datasetType, const std::vector<int>& kValues, std::vector<double>& mapScores);
    static std::string getImageClass(const std::string& filename, int datasetType, bool queryfix = true);
};

#endif