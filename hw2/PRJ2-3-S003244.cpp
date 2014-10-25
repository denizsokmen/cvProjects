/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>


using namespace cv;
using namespace std;

Mat getHistogram(Mat& image, int quantum) {
    int numBins = 256/quantum;

    Mat dest(100, numBins * image.channels(), CV_32FC1, cvScalar(255));
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

    Mat dzt(500, numBins * image.channels(), CV_8UC3, cvScalar(255, 255, 255));
    fprintf(stderr, "%d\n", dest.cols);
    for (int i = 0; i < dest.cols-1; i++) {
        float scalar = (float)dzt.rows / maxcol;
        float mult = dzt.rows - scalar*(bins[i] * sumofall);
        float mult2 = dzt.rows - scalar*(bins[i+1] * sumofall);

        Vec3b newcol(((i / numBins) == 0) ? 255 : 0, ((i / numBins) == 1) ? 255 : 0, ((i / numBins) == 2) ? 255 : 0);
        line(dzt, Point(i, mult), Point(i+1, mult2), Scalar(newcol), 2, 8, 0);

       // dzt.at<Vec3b>((int)mult, i) = newcol;
       // dest.at<float>(mult, i) = bins[i];

    }
    return dzt;
}

int getdir (string dir, map<string, vector<string>> &files, string category)
{
    DIR *dp;
    struct stat statbuff;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL) {
        //cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        stat(dirp->d_name, &statbuff);
        if (dirp->d_type & DT_DIR) {
            char path[1024];
            int len = snprintf(path, sizeof(path)-1, "%s/%s", dir.c_str(), dirp->d_name);
            path[len] = 0;
            if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
                continue;

            getdir(path, files, string(dirp->d_name));

        }
        else {
            if (strcmp(category.c_str(), ".") == 0)
                continue;

            files[category].push_back(string(dirp->d_name));
        }
    }
    closedir(dp);
    return 0;
}

int main(int argc, char** argv )
{
    string dir = string(".");
    map<string, vector<string>> files = map<string, vector<string>>();

    getdir(dir,files, ".");

    for (auto it = files.begin(); it != files.end(); ++it) {
        printf("%s \n", it->first.c_str());
        for (auto itvec = it->second.begin(); itvec != it->second.end(); itvec++) {
            printf(" - %s\n", itvec->c_str());
        }
    }
    Mat image;
    image = imread( "lena.jpg", 1 );


    namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
    Mat result = getHistogram(image, 2);
    imshow("Display Image", result);
    waitKey(0);


    return 0;
}