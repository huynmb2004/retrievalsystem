#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

#include "FeatureExtractor.h"
#include "ColorHistogram.h"
#include "ColorCorrelogram.h"
#include "SIFTExtractor.h"
#include "ORBExtractor.h"
#include "DatabaseManager.h"
#include "TextureFeature.h"
#include "EdgeFeatureExtractor.h"
#include "CombinedFeature.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;
using namespace std::chrono;

// Global variables for UI
string queryImagePath = "C:/Users/huyng/Downloads/TMBuD-main/TMBuD-main/query/007.png";
string galleryPath = "C:/Users/huyng/Downloads/TMBuD-main/TMBuD-main/images";
string method = "ColorHistogram";
vector<pair<string, double>> results;
DatabaseManager* dbManager = nullptr;
FeatureExtractor* extractor = nullptr;
const string databaseDir = "build/database/";
bool showResultsFlag = false;
bool showMAPResultsFlag = false;
Mat currentResultsDisplay; 
bool mapResultsDisplayed = false;
bool resultsWindowOpen = false;
int datasetType = 0; // 1: TMBuD

// Function prototypes
void createDatabase();
void performQuery(const std::string& queryImagePath);
void showResults();
void onTrackbar(int, void*);
Mat createResultsDisplay(double queryTimeMs = 0);
unique_ptr<FeatureExtractor> createCombinedExtractor(const vector<int>& methods);
void showMAPResults(const vector<double>& mapScores, const vector<int>& kValues);

// UI Main function
int main(int argc, char** argv) {
    // Create database directory if not exists
    fs::create_directories(databaseDir);

    // Create main window
    namedWindow("Image Retrieval System", WINDOW_NORMAL);
    resizeWindow("Image Retrieval System", 1200, 800);

    // Create UI elements
    int methodSelection = 0;
    createTrackbar("Method", "Image Retrieval System", &methodSelection, 7, onTrackbar);

    // Main loop
    while (true) {
        Mat im = Mat::zeros(800, 1200, CV_8UC3);
        
        // Show instructions
        putText(im, "1. Press 'q' to select query image", Point(50, 50), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(im, "2. Press 'g' to select gallery folder", Point(50, 100), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(im, "3. Press 'r' to run retrieval", Point(50, 150), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(im, "4. Press ESC to exit", Point(50, 200), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        
        // Show current selections
        putText(im, "Query: " + queryImagePath, Point(50, 350), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        putText(im, "Gallery: " + galleryPath, Point(50, 400), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        putText(im, "Method: " + method, Point(50, 450), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        imshow("Image Retrieval System", im);
        
        // Handle key presses - only if main window has focus
        int key = waitKey(30);
        if (getWindowProperty("Image Retrieval System", WND_PROP_VISIBLE) >= 1) {
            if (key == 27) {
                cout << "Exiting program..." << endl;
                break;
            } // ESC
            else if (key == 'q') {
                cout << "Enter path to query image: ";
                string path;
                cin >> path;
                if (!path.empty()) queryImagePath = path;
            }
            else if (key == 'g') {
                cout << "Enter path to gallery image: ";
                string path;
                cin >> path;
                if (!path.empty()) galleryPath = path;
                // String path = "./assets/training_set/training_images/";
                // galleryPath = path;
            }
            else if (key == 'r') {
                if (queryImagePath.empty() || galleryPath.empty()) {
                    cout << "Please select both query image and gallery folder first!" << endl;
                } else {
                    // C:/Users/huyng/Downloads/TMBuD-main/TMBuD-main/images
                    cout << galleryPath.substr(galleryPath.size() - 17) << endl;
                    if (galleryPath.substr(galleryPath.size() - 17) == "TMBuD-main/images") {
                        datasetType = 1; // TMBuD dataset
                    }
                    createDatabase();
                    performQuery(queryImagePath);
                    
                    // Show Map (handled inside performQuery)
                    mapResultsDisplayed = true;
                    
                    // Show new results
                    showResults();
                    resultsWindowOpen = true;
                }
            }
        } else {
            break; // Exit if main window is closed
        }
        
        if(mapResultsDisplayed && getWindowProperty("MAP Evaluation", WND_PROP_VISIBLE) < 1) {
            mapResultsDisplayed = false;
        }

        // Check if results window was closed by user
        if (resultsWindowOpen && getWindowProperty("Retrieval Results", WND_PROP_VISIBLE) < 1) {
            resultsWindowOpen = false;
        }
    }

    cout << "Method: " << method << endl;
    cout << "Extractor: " << (extractor ? extractor->getMethodName() : "NULL") << endl;
    cout << "DB Manager: " << (dbManager ? "Exists" : "NULL") << endl;

    // Clean up
    if (dbManager) delete dbManager;
    if (extractor) delete extractor;
    destroyAllWindows();
    return 0;
}

void onTrackbar(int val, void*) {
    switch (val) {
        case 0: method = "ColorHistogram"; break;
        case 1: method = "ColorCorrelogram"; break;
        case 2: method = "SIFT"; break;
        case 3: method = "ORB"; break;
        case 4: method = "Combined_ColorHist+Edge"; break;
        case 5: method = "Combined_ColorHist+SIFT"; break;
        case 6: method = "Combined_SIFT+Edge"; break;
        case 7: method = "Combined_ColorHist+SIFT+Edge"; break;
    }
}

void createDatabase() {

    try {
        // Create the appropriate feature extractor based on selected method
        if (method == "ColorHistogram") {
            extractor = new ColorHistogram(8, true); // 8 bins, using HSV
        } 
        else if (method == "ColorCorrelogram") {
            extractor = new ColorCorrelogram(8, {1, 3, 5}); // 8 bins, distances 1,3,5
        }
        else if (method == "SIFT") {
            extractor = new SIFTExtractor(500, 3); // 500 features, 3 octave layers
        }
        else if (method == "ORB") {
            extractor = new ORBExtractor(1000); // 500 features
        }
        else if (method == "Texture_LBP") {
            extractor = new TextureFeature(); // Default constructor
        }
        else if (method == "Edge") {
            extractor = new EdgeFeatureExtractor(); // Default constructor
        }
        else if (method == "Combined_ColorHist+Edge") {
            auto combined = createCombinedExtractor({0, 5});
            if (!combined) {
                throw runtime_error("Failed to create combined feature extractor");
            }
            extractor = combined.release();
        }
        else if (method == "Combined_SIFT+Edge") {
            auto combined = createCombinedExtractor({2, 5});
            if (!combined) {
                throw runtime_error("Failed to create combined feature extractor");
            }
            extractor = combined.release();
        }
        else if (method == "Combined_ColorHist+SIFT") {
            auto combined = createCombinedExtractor({0, 2});
            if (!combined) {
                throw runtime_error("Failed to create combined feature extractor");
            }
            extractor = combined.release();
        }
        else if (method == "Combined_ColorHist+SIFT+Edge") {
            auto combined = createCombinedExtractor({0, 2, 5});
            if (!combined) {
                throw runtime_error("Failed to create combined feature extractor");
            }
            extractor = combined.release();
        }
        else {
            throw runtime_error("Unknown method: " + method);
        }

        // Verify extractor was created
        if (!extractor) {
            throw runtime_error("Feature extractor creation failed for method: " + method);
        }

        // Database file path
        string dbPath = DatabaseManager::getDatabasePath(method, galleryPath);
        
        // Check if database exists
        if (fs::exists(dbPath)) {
            cout << "Loading existing database for method: " << method << endl;
            if (dbManager) {
                delete dbManager;
            }
            dbManager = new DatabaseManager(extractor);
            dbManager->loadDatabase(dbPath);
            cout << "Database loaded successfully with " 
                 << dbManager->getDatabaseSize() << " entries." << endl;
        } 
        else {
            cout << "Creating new database for method: " << method << endl;
            
            // Get all valid image paths from gallery
            vector<string> imagePaths;
            int skipped = 0;
            for (const auto& entry : fs::directory_iterator(galleryPath)) {
                string path = entry.path().string();
                
                // Check if the file is an image
                string ext = entry.path().extension().string();
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                    // Verify the image can be loaded
                    Mat testImage = imread(path);
                    if (testImage.empty()) {
                        cerr << "Warning: Could not read image " << path << " - skipping" << endl;
                        skipped++;
                        continue;
                    }
                    imagePaths.push_back(path);
                }
            }

            if (imagePaths.empty()) {
                throw runtime_error("No valid images found in gallery path: " + galleryPath);
            }

            cout << "Found " << imagePaths.size() << " valid images (" 
                 << skipped << " files skipped)" << endl;

            // Create and save database
            if (dbManager) {
                delete dbManager;
            }
            dbManager = new DatabaseManager(extractor);
            
            cout << "Building database..." << endl;
            dbManager->buildDatabase(imagePaths);
            
            cout << "Saving database to " << dbPath << endl;
            dbManager->saveDatabase(dbPath);
            
            cout << "Database created successfully with " 
                 << dbManager->getDatabaseSize() << " entries." << endl;
        }
    } 
    catch (const exception& e) {
        cerr << "Error in createDatabase(): " << e.what() << endl;
        
        // Clean up on error
        if (extractor) {
            delete extractor;
            extractor = nullptr;
        }
        if (dbManager) {
            delete dbManager;
            dbManager = nullptr;
        }
        throw; // Re-throw to let caller handle the error
    }
}

void performQuery(const std::string& queryImagePath) {
    cout << "Starting query with method: " << method << endl;
    
    if (!dbManager || queryImagePath.empty()) {
        cout << "Database manager not initialized or no query path provided!" << endl;
        return;
    }
    
    Mat queryImage = imread(queryImagePath);
    if (queryImage.empty()) {
        cout << "Could not load query image!" << endl;
        return;
    }

    try {
        auto t1 = high_resolution_clock::now();


        // Perform query using the database manager
        vector<int> kValues = {3, 5, 11, 21};
        vector<double> mapScores;
        
        // Show MAP results
        cout << "Executing queryWithMAP..." << endl;
        results = dbManager->queryWithMAP(queryImage, queryImagePath, datasetType , kValues, mapScores);
        
        cout << "Showing MAP results..." << endl;
        showMAPResults(mapScores, kValues);
        showMAPResultsFlag = true;

        // Get top results
        cout << "Getting top results..." << endl;
        results = dbManager->query(queryImage, 12);

        auto t2 = high_resolution_clock::now();
        double queryTimeMs = static_cast<double>(duration_cast<milliseconds>(t2 - t1).count());
        
        cout << "Displaying results..." << endl;
        currentResultsDisplay = createResultsDisplay(queryTimeMs);
        showResultsFlag = true;
    } catch (const exception& e) {
        cerr << "Query failed: " << e.what() << endl;
    }
}

// Create a combined feature extractor with specified methods
unique_ptr<FeatureExtractor> createCombinedExtractor(const vector<int>& methods) {

    try {
        vector<unique_ptr<FeatureExtractor>> extractors;
        vector<double> weights(methods.size(), 1.0 / methods.size()); // Equal weights for simplicity
        for (int method : methods) {
            switch (method) {
                case 0: // ColorHistogram
                    extractors.push_back(make_unique<ColorHistogram>(8, true));
                    break;
                case 1: // ColorCorrelogram
                    extractors.push_back(make_unique<ColorCorrelogram>(8, std::vector<int>{1, 3, 5}));
                    break;
                case 2: // SIFT
                    extractors.push_back(make_unique<SIFTExtractor>(500, 3));
                    break;
                case 3: // ORB
                    extractors.push_back(make_unique<ORBExtractor>(1000));
                    break;
                case 4: // Texture_LBP
                    extractors.push_back(make_unique<TextureFeature>());
                    break;
                case 5: // EdgeFeatureExtractor
                    extractors.push_back(make_unique<EdgeFeatureExtractor>());
                    break;
                default:
                    throw runtime_error("Unknown feature extractor method: " + to_string(method));
            }
        }

        if (methods[0] == 0 && methods.size() == 2) {
            // Special case for Combined_ColorHist+Edge+ORB
            weights = {0.15, 0.85};
        } else if (methods[0] == 0 && methods.size() == 3) {
            // Special case for Combined_ColorHist+Edge
            weights = {0.2, 0.5, 0.3};
        }
        else if (methods[0] == 2 && methods.size() == 2) {
            // Special case for Combined_SIFT+Edge
            weights = {0.7, 0.3};
        }
        
        return make_unique<CombinedFeature>(move(extractors), weights);
    } catch (const exception& e) {
        cerr << "Error creating combined extractor: " << e.what() << endl;
        return nullptr;
    }
}


// Create a display image for results
void showResults() {
    if (results.empty() || queryImagePath.empty()) return;

    // Load query image
    Mat queryImage = imread(queryImagePath);
    if (queryImage.empty()){
        cout << "Could not load query image: " << queryImagePath << endl;
        return;
    }
    
    if (!currentResultsDisplay.empty()) {
        namedWindow("Retrieval Results", WINDOW_NORMAL);
        imshow("Retrieval Results", currentResultsDisplay);

        // wait for user to close the window or press ESC
        while (getWindowProperty("Retrieval Results", WND_PROP_VISIBLE) >= 1) {
            int key = waitKey(30);
            if (key == 27) { // ESC
                destroyWindow("Retrieval Results");
                break;
            }
        }
        return;
    }
    
    // Constants for display
    const int thumbWidth = 200;  
    const int thumbHeight = 200; 
    const int margin = 10;       
    const int textHeight = 30;   

    // Calculate display dimensions
    int cols = 3; 
    int rows = (results.size() + cols - 1) / cols; 
    
    // Create result window - size based on content
    int displayWidth = cols * (thumbWidth + margin) + margin;
    int displayHeight = rows * (thumbHeight + textHeight + margin) + margin + thumbHeight + margin;
    
    // Create a big image to display all results
    Mat displayImage = Mat::zeros(displayHeight, displayWidth, CV_8UC3);
    displayImage.setTo(Scalar(40, 40, 40)); 

    // Show window
    // Hiển thị cửa sổ kết quả
    namedWindow("Retrieval Results", WINDOW_NORMAL);
    resizeWindow("Retrieval Results", displayWidth, displayHeight);
    imshow("Retrieval Results", displayImage);
    
    // Chờ cho đến khi cửa sổ kết quả bị đóng
    while (getWindowProperty("Retrieval Results", WND_PROP_VISIBLE) >= 1) {
        // Chỉ xử lý phím ESC để thoát hoàn toàn
        int key = waitKey(30);
        if (key == 27) { 
            destroyWindow("Retrieval Results");
            break;
        }
    }
}


Mat createResultsDisplay(double queryTimeMs) {
    if (results.empty() || queryImagePath.empty()) return Mat();

    // Load query image
    Mat queryImage = imread(queryImagePath);
    if (queryImage.empty()) {
        cout << "Could not load query image: " << queryImagePath << endl;
        return Mat();
    }

    // Layout constants
    const int thumbWidth = 200;
    const int thumbHeight = 200;
    const int margin = 30;
    const int textHeight = 30;
    const int candidateCols = 3;
    int candidateRows = (results.size() + candidateCols - 1) / candidateCols;

    // Calculate display size
    int leftPanelWidth = thumbWidth + 2 * margin;
    int rightPanelWidth = candidateCols * (thumbWidth + margin) + margin;
    int displayWidth = leftPanelWidth + rightPanelWidth;
    int displayHeight = max(
        margin + thumbHeight + textHeight + margin, // query image panel
        margin + candidateRows * (thumbHeight + textHeight + margin) // candidate panel
    ) + margin + 60; // extra for top bar

    Mat displayImage = Mat::zeros(displayHeight, displayWidth, CV_8UC3);
    displayImage.setTo(Scalar(40, 40, 40));

    // Draw method name (center top)
    string methodText = "Method: " + method;
    int methodTextWidth = getTextSize(methodText, FONT_HERSHEY_SIMPLEX, 1.0, 2, 0).width;
    putText(displayImage, methodText, 
        Point((displayWidth - methodTextWidth) / 2, 45), 
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0), 2);

    // Draw query image (left panel)
    Mat queryDisplay;
    resize(queryImage, queryDisplay, Size(thumbWidth, thumbHeight), 0, 0, INTER_AREA);
    int queryX = margin;
    int queryY = margin + 60; // leave space for top bar
    copyMakeBorder(queryDisplay, queryDisplay, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(255, 255, 255));
    queryDisplay.copyTo(displayImage(Rect(queryX, queryY, thumbWidth + 4, thumbHeight + 4)));

    // Draw "Query image" label
    string queryLabel = "Query image";
    int queryLabelWidth = getTextSize(queryLabel, FONT_HERSHEY_SIMPLEX, 0.8, 2, 0).width;
    putText(displayImage, queryLabel, 
        Point(queryX + (thumbWidth + 4 - queryLabelWidth) / 2, queryY + thumbHeight + 40), 
        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

     if (queryTimeMs >= 0) {
        string timeText = "Query time: " + to_string((int)queryTimeMs) + " ms";
        putText(displayImage, timeText, Point(queryX + (thumbWidth + 4 - queryLabelWidth) - 100 / 2, queryY + thumbHeight + 80),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
    }

    // Draw candidate images (right panel)
    int startX = leftPanelWidth;
    int startY = margin + 60;
    for (size_t i = 0; i < results.size(); i++) {
        Mat resultImage = imread(results[i].first);
        if (resultImage.empty()) continue;

        int col = i % candidateCols;
        int row = i / candidateCols;
        int x = startX + margin + col * (thumbWidth + margin);
        int y = startY + row * (thumbHeight + textHeight + margin);

        Mat resizedResult;
        double aspect = (double)resultImage.cols / resultImage.rows;
        int newWidth, newHeight;
        if (aspect > 1.0) {
            newWidth = thumbWidth;
            newHeight = cvRound(thumbWidth / aspect);
        } else {
            newHeight = thumbHeight;
            newWidth = cvRound(thumbHeight * aspect);
        }
        resize(resultImage, resizedResult, Size(newWidth, newHeight), 0, 0, INTER_AREA);

        int xOffset = (thumbWidth - newWidth) / 2;
        int yOffset = (thumbHeight - newHeight) / 2;

        Mat borderedResult;
        copyMakeBorder(resizedResult, borderedResult,
            yOffset, thumbHeight - newHeight - yOffset,
            xOffset, thumbWidth - newWidth - xOffset,
            BORDER_CONSTANT, Scalar(50, 50, 50));

        borderedResult.copyTo(displayImage(Rect(x, y, thumbWidth, thumbHeight)));

        // Draw distance below each candidate
        stringstream info;
        info << fixed << setprecision(2) << results[i].second;
        string distText = "Dist: " + info.str();
        int textWidth = getTextSize(distText, FONT_HERSHEY_SIMPLEX, 0.7, 2, 0).width;
        putText(displayImage, distText, 
            Point(x + (thumbWidth - textWidth) / 2, y + thumbHeight + 30), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 255), 2);
    }

    return displayImage;
}

void showMAPResults(const vector<double>& mapScores, const vector<int>& kValues) {
    Mat mapDisplay = Mat::zeros(300, 500, CV_8UC3);
    mapDisplay.setTo(Scalar(240, 240, 240));
    
    putText(mapDisplay, "MAP Evaluation Results", Point(50, 30), 
           FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
    
    for (size_t i = 0; i < mapScores.size(); ++i) {
        string text = "MAP@ " + to_string(kValues[i]) + ": " + to_string(mapScores[i]);
        putText(mapDisplay, text, Point(50, 80 + i * 30), 
               FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1.5);
    }
    
    namedWindow("MAP Evaluation", WINDOW_NORMAL);
    imshow("MAP Evaluation", mapDisplay);
    
    while (getWindowProperty("MAP Evaluation", WND_PROP_VISIBLE) >= 1) {
        // Chỉ xử lý phím ESC để thoát hoàn toàn
        int key = waitKey(30);
        if (key == 27) { 
            destroyWindow("MAP Evaluation");
            break;
        }
    }
}