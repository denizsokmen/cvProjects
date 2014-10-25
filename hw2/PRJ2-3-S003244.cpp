/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>


using namespace cv;
using namespace std;

Mat getHistogram(Mat& image, int quantum) {
    Mat dest(100, (256 / quantum) * image.channels(), CV_32FC1, cvScalar(255));
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
                int bin = (256 * k) + (illumination / quantum);
                bins[bin] += 1.0f;
                sumofall += 1.0;
                maxcol = max(maxcol, bins[bin]);
                mincol = min(mincol, bins[bin]);
            }
        }
    }

    for(auto it = bins.begin(); it != bins.end(); ++it) {
        it->second /= sumofall;
        //fprintf(stderr, "%d - %f\n", it->first, it->second);
    }

    Mat dzt(500, (256 / quantum) * image.channels(), CV_8UC3, cvScalar(255, 255, 255));
    fprintf(stderr, "%d\n", dest.cols);
    for (int i = 0; i < dest.cols-1; i++) {
        float scalar = (float)dzt.rows / maxcol;
        float mult = dzt.rows - scalar*(bins[i] * sumofall);
        float mult2 = dzt.rows - scalar*(bins[i+1] * sumofall);

        fprintf(stderr, "%f\n", mult);
        Vec3b newcol(((i / 256) == 0) ? 255 : 0, ((i / 256) == 1) ? 255 : 0, ((i / 256) == 2) ? 255 : 0);
        line(dzt, Point(i, mult), Point(i+1, mult2), Scalar(newcol), 2, 8, 0);

       // dzt.at<Vec3b>((int)mult, i) = newcol;
       // dest.at<float>(mult, i) = bins[i];

    }
    return dzt;
}

int getdir (string dir, vector<string> &files, int lvl)
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

            //char filedir[1024];
            //sprintf(filedir, "%*s[%s]", lvl*2, "", dirp->d_name);
            //printf("%*s[%s]\n", lvl, "", dirp->d_name);
            //files.push_back(string(filedir));
            files.push_back(string(path));
            getdir(path, files, lvl++);

        }
        else {
            char filedir[1024];
            sprintf(filedir, "%s/%s", dir.c_str(), dirp->d_name);
            files.push_back(string(filedir));
        }
    }
    closedir(dp);
    return 0;
}

int main(int argc, char** argv )
{
    string dir = string(".");
    vector<string> files = vector<string>();

    getdir(dir,files, 0);

    //for (unsigned int i = 0;i < files.size();i++) {
    //    cout << files[i] << endl;
    //}

    Mat image;
    image = imread( "lena.jpg", 1 );


    namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
    Mat result = getHistogram(image, 1);
    imshow("Display Image", result);
    waitKey(0);


    return 0;
}