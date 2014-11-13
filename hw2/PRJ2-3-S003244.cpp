/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>


using namespace cv;
using namespace std;


map<string, vector<Mat>> trainingSet = map<string, vector<Mat>>();
map<string, vector<Mat>> testSet = map<string, vector<Mat>>();

Mat getHistogram(Mat& image, int quantum, int grid) {
    int numBins = 256/quantum;
    int totalBins = image.channels() * numBins;
    int gridSquare = grid*grid;

    Mat dest(1, totalBins * gridSquare, CV_32FC1, cvScalar(255));
    map<int, float> bins = map<int, float>();

    double* sumofall = new double[gridSquare];
    for (int i = 0; i < dest.cols; i++) {
        bins[i] = 0.0f;
    }


    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Vec3b col = image.at<Vec3b>(i, j);
            int offsetRow = min(grid-1, (i-1) / (image.rows / grid));
            int offsetCol = min(grid-1, (j-1) / (image.cols / grid));
            int offset = offsetCol * grid + offsetRow;

            for (int k = 0; k < image.channels(); k++) {
                int illumination = col.val[k];
                int bin = (totalBins*offset) + (numBins * k) + (illumination / quantum);
                bins[bin] += 1.0f;
                sumofall[offset] += 1.0;
            }
        }
    }

    for(auto it = bins.begin(); it != bins.end(); ++it) {
        int offset = it->first / totalBins;
        it->second /= sumofall[offset];
    }

    for (int i = 0; i < dest.cols-1; i++) {
        dest.at<float>(0, i) = bins[i];
    }

    return dest;
}

int getNumOfJPG(string dir) {
    int len;
    struct dirent *pdir;
    DIR *pDir;
    int count = 0;

    pDir = opendir(dir.c_str());

    if (pDir != NULL) {
        while ((pdir = readdir(pDir)) != NULL) {
            len = strlen(pdir->d_name);
            if (len >= 4) {
                if (strcmp(".jpg", &(pdir->d_name[len - 4])) == 0) {
                    count++;
                }
            }
        }
    }
    closedir (pDir);

    return count;
}

int traverseImages(string dir, map<string, vector<Mat>> &trainingFiles, map<string, vector<Mat>> &testFiles, string category, int count)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL) {
        //cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if (dirp->d_type & DT_DIR) {
            char path[1024];
            int len = snprintf(path, sizeof(path)-1, "%s/%s", dir.c_str(), dirp->d_name);
            path[len] = 0;
            if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
                continue;

            traverseImages(path, trainingFiles, testFiles, string(dirp->d_name), getNumOfJPG(path));

        }
        else {
            if (strcmp(category.c_str(), ".") == 0)
                continue;

            int len = strlen(dirp->d_name);

            if (len >= 4) {
                if (strcmp(".jpg", dirp->d_name + len - 4) == 0) {
                    char path[1024];
                    int len = snprintf(path, sizeof(path)-1, "%s/%s", dir.c_str(), dirp->d_name);

                    Mat histogram = imread(path, 1);
                    histogram = getHistogram(histogram, 4, 1);
                    if (trainingFiles[category].size() < count / 2)
                        trainingFiles[category].push_back(histogram);
                    else
                        testFiles[category].push_back(histogram);
                }
            }
        }
    }
    closedir(dp);
    return 0;
}

double intersectHist(Mat& h1, Mat& h2) {
    double sum = 0.0f;

    for (int i = 0; i < h1.cols; i++) {
        double h1val = h1.at<float>(0, i);
        double h2val = h2.at<float>(0, i);
        sum += min(h1val, h2val);
        //sum += (h1val - h2val) * (h1val-h2val) / (h1val + h2val);
        //sum += (h1.at<float>(0, i) - h2.at<float>(0, i)) * (h1.at<float>(0, i) - h2.at<float>(0, i)) / (h1.at<float>(0, i) + h2.at<float>(0, i));
    }

    return 1.0-sum;
}

int main(int argc, char** argv )
{
    string dir = string(".");
    traverseImages(dir, trainingSet, testSet, ".", 0);

    namedWindow("Display Image", CV_WINDOW_AUTOSIZE );

    double minDist = 1.0;
    string minCategory = " ";
    map<string, vector<string>> results = map<string, vector<string>>();


    // map the training category with test category
    for (auto testIt = testSet.begin(); testIt != testSet.end(); ++testIt) {
        for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
            for (auto it = trainingSet.begin(); it != trainingSet.end(); ++it) {
                for (auto trainingImg = it->second.begin(); trainingImg != it->second.end(); trainingImg++) {
                    double res = intersectHist(*trainingImg, *testImg);

                    if (res < minDist) {
                        minDist = res;
                        minCategory = it->first;
                    }
                }
            }

            results[minCategory].push_back(testIt->first);
            minDist = 1.0f;
        }
    }

    int matchedSize = 0;
    int testSize = 0;

    for (auto testIt = results.begin(); testIt != results.end(); ++testIt) {
        matchedSize = 0;
        testSize = 0;
        fprintf(stderr, "%s - ", testIt->first.c_str());
        testSize += testSet[testIt->first].size();
        for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
            if (*testImg == testIt->first)
                matchedSize++;
        }
        fprintf(stderr, "%f%\n", (float)100.0f*matchedSize / testSize);
    }
    //fprintf(stderr, "Overall: %f%\n", (float)100.0f*matchedSize / testSize);


    return 0;
}