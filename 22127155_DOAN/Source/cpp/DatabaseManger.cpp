#include "DatabaseManager.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

using namespace std;
using namespace cv;

DatabaseManager::DatabaseManager(FeatureExtractor* extractor) 
    : extractor(extractor) {}

DatabaseManager::~DatabaseManager() {
    delete extractor;
}

std::string DatabaseManager::getDatabasePath(const std::string& method, const std::string& datasetPath) {
    // Lấy tên thư mục cuối cùng của datasetPath
    std::filesystem::path dsPath(datasetPath);
    // Nếu muốn chắc chắn không trùng, có thể dùng hash:
    std::size_t hashVal = std::hash<std::string>{}(datasetPath);
    return "build/database/" + method + "_" + std::to_string(hashVal) + "_features.csv";
    // return "build/database/" + method + "_" + datasetName + "_features.csv";
}

void DatabaseManager::buildDatabase(const vector<string>& imagePaths) {
    cout << "Building database" << endl;
    featuresDB.clear();
    for (const auto& path : imagePaths) {
        Mat image = imread(path);
        if (image.empty()) {
            std::cerr << "[DEBUG] imread failed for: " << path << std::endl;
            continue;
        }
        
        vector<float> features = extractor->extract(image);
        featuresDB[path] = features;
    }
}

void DatabaseManager::saveDatabase(const string& filePath) {
    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());
    ofstream outFile(filePath);
    if (!outFile.is_open()) {
        cerr << "Error opening file for writing: " << filePath << endl;
        return;
    }
    
    // Ghi header
    outFile << "image_path,feature_method,features\n";
    
    // Ghi dữ liệu
    for (const auto& entry : featuresDB) {
        outFile << entry.first << ","
                << extractor->getMethodName() << ","
                << extractor->featuresToString(entry.second) << "\n";
    }
    
    outFile.close();
}

bool DatabaseManager::loadDatabase(const string& filePath) {
    ifstream inFile(filePath);
    if (!inFile.is_open()) {
        cerr << "Error opening file for reading: " << filePath << endl;
        return false;
    }
    
    featuresDB.clear();
    string line;
    
    // Bỏ qua header
    if (!getline(inFile, line)) {
        inFile.close();
        return false;
    }
    
    while (getline(inFile, line)) {
        size_t pos1 = line.find(',');
        size_t pos2 = line.find(',', pos1 + 1);
        
        if (pos1 == string::npos || pos2 == string::npos) continue;
        
        string path = line.substr(0, pos1);
        string method = line.substr(pos1 + 1, pos2 - pos1 - 1);
        string featureStr = line.substr(pos2 + 1);
        
        // Chỉ đọc nếu phương pháp trích xuất phù hợp
        if (method == extractor->getMethodName()) {
            featuresDB[path] = extractor->stringToFeatures(featureStr);
        }
    }
    
    inFile.close();
    return true;
}

vector<pair<string, double>> DatabaseManager::query(const Mat& queryImage, int topK) {
    cout << "Querying database for image" << endl;
    vector<float> queryFeatures = extractor->extract(queryImage);
    vector<pair<string, double>> results;
    
    for (const auto& entry : featuresDB) {
        double distance = extractor->compare(queryFeatures, entry.second);
        results.emplace_back(entry.first, distance);
    }
    
    // Sắp xếp theo khoảng cách (tăng dần)
    sort(results.begin(), results.end(), 
        [](const pair<string, double>& a, const pair<string, double>& b) {
            return a.second < b.second;
        });
    
    // Giới hạn số lượng kết quả
    if (topK > 0 && results.size() > topK) {
        results.resize(topK);
    }
    
    return results;
}

std::string DatabaseManager::getExtractorName() const {
    return extractor->getMethodName();
}

size_t DatabaseManager::getDatabaseSize() const {
    return featuresDB.size();
}

void DatabaseManager::setExtractor(FeatureExtractor* newExtractor) {
    delete extractor;
    extractor = newExtractor;
}

std::string DatabaseManager::getImageClass(const std::string& filename, int datasetType, bool queryfix) {
    if (datasetType == 1) {
        return filename.substr(filename.length()-9, 3);
        // return queryfix ? filename.substr(filename.length()-7, 3) : filename.substr(filename.length()-9, 3);
    } else {
        return queryfix ? filename.substr(filename.length()-6, 2) : filename.substr(filename.length()-8, 2);
    }
}

vector<pair<string, double>> DatabaseManager::queryWithMAP(const Mat& queryImage, 
                                                         const string& queryImagePath, int datasetType,
                                                         const vector<int>& kValues, 
                                                         vector<double>& mapScores) {
    cout << "Querying with MAP for image" << endl;
    vector<float> queryFeatures = extractor->extract(queryImage);
    vector<pair<string, double>> allResults;
    
    // Tính toán kết quả cho tất cả ảnh
    for (const auto& entry : featuresDB) {
        double distance = extractor->compare(queryFeatures, entry.second);
        allResults.emplace_back(entry.first, distance);
    }
    
    // Sắp xếp theo khoảng cách tăng dần
    sort(allResults.begin(), allResults.end(), 
        [](const pair<string, double>& a, const pair<string, double>& b) {
            return a.second < b.second;
        });

    // 4. Đếm tổng số ảnh liên quan (trong toàn bộ DB)
    std::string queryClass = getImageClass(queryImagePath, datasetType);

    int totalRelevant = 0;
    for (const auto& entry : featuresDB) {
        if (getImageClass(entry.first, datasetType, false) == queryClass)
            totalRelevant++;
    }
    
    // Tính MAP cho các giá trị k khác nhau
    mapScores.clear();
    for (int k : kValues) {
        cout << "Calculating MAP for k = " << k << endl;
        double ap = 0.0;
        int hit = 0;
        
        for (int i = 0; i < min(k, (int)allResults.size()); ++i) {
            std::string resultClass = getImageClass(allResults[i].first, datasetType, false);
            if (resultClass == queryClass) {
                hit++;
                ap += (double)hit / (i + 1);
            }
            cout << "Hit: " << hit << ", AP: " << ap << endl;
        }
        
        if (totalRelevant > 0) {
            ap /= totalRelevant;
        }
        mapScores.push_back(ap);
    }
    
    return allResults;
}