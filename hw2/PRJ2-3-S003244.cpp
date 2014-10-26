/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>


using namespace cv;
using namespace std;


map<string, vector<Mat>> trainingSet = map<string, vector<Mat>>();
map<string, vector<Mat>> testSet = map<string, vector<Mat>>();

Mat getHistogram(Mat& image, int quantum) {
    int numBins = 256/quantum;

    Mat dest(1, numBins * image.channels(), CV_32FC1, cvScalar(255));
    map<int, float> bins = map<int, float>();

    float maxcol = 0;
    float mincol = 0;

    double sumofall = 0.0;
    for (int i = 0; i < dest.cols; i++) {
        bins[i] = 0.0f;
    }



    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Vec3b col = image.at<Vec3b>(i, j);
            for (int k = 0; k < image.channels(); k++) {
                int illumination = col.val[k];
                int bin = (numBins * k) + (illumination / quantum);
                bins[bin] += 1.0f;
                sumofall += 1.0;
                maxcol = max(maxcol, bins[bin]);
                mincol = min(mincol, bins[bin]);
            }
        }
    }

    for(auto it = bins.begin(); it != bins.end(); ++it) {
        it->second /= sumofall;
    }

    //Mat dzt(500, numBins * image.channels(), CV_8UC3, cvScalar(255, 255, 255));
    for (int i = 0; i < dest.cols-1; i++) {
        /* for testing purposes
        float scalar = (float)dzt.rows / maxcol;
        float mult = dzt.rows - scalar*(bins[i] * sumofall);
        float mult2 = dzt.rows - scalar*(bins[i+1] * sumofall);

        Vec3b newcol(((i / numBins) == 0) ? 255 : 0, ((i / numBins) == 1) ? 255 : 0, ((i / numBins) == 2) ? 255 : 0);
        line(dzt, Point(i, mult), Point(i+1, mult2), Scalar(newcol), 2, 8, 0);*/

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

int getdir (string dir, map<string, vector<Mat>> &trainingFiles, map<string, vector<Mat>> &testFiles, string category, int count)
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

            getdir(path, trainingFiles, testFiles, string(dirp->d_name), getNumOfJPG(path));

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
                    histogram = getHistogram(histogram, 8);
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

float intersectHist(Mat& h1, Mat& h2) {
    float sum = 0.0f;

    for (int i = 0; i < h1.cols; i++) {
        sum += (h1.at<float>(0, i) - h2.at<float>(0, i)) * (h1.at<float>(0, i) - h2.at<float>(0, i)) / (h1.at<float>(0, i) + h2.at<float>(0, i));
    }

    return sum/2;
}

int main(int argc, char** argv )
{
    string dir = string(".");

    getdir(dir, trainingSet, testSet, ".", 0);

    namedWindow("Display Image", CV_WINDOW_AUTOSIZE );

    float minDist = 1.0f;
    string minCategory = " ";
    map<string, vector<string>> results = map<string, vector<string>>();


    for (auto testIt = testSet.begin(); testIt != testSet.end(); ++testIt) {
        for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
            for (auto it = trainingSet.begin(); it != trainingSet.end(); ++it) {
                for (auto trainingImg = it->second.begin(); trainingImg != it->second.end(); trainingImg++) {
                    float res = intersectHist(*trainingImg, *testImg);

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

    for (auto testIt = results.begin(); testIt != results.end(); ++testIt) {
        fprintf(stderr, "%s - ", testIt->first.c_str());
        int matchedSize = 0;
        int testSize = testSet[testIt->first].size();
        for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
            if (*testImg == testIt->first)
                matchedSize++;
        }
        fprintf(stderr, "%f%\n", (float)100.0f*matchedSize / testSize);
    }


    return 0;
}