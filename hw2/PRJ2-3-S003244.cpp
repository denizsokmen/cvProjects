/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>


using namespace cv;
using namespace std;

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

    for (unsigned int i = 0;i < files.size();i++) {
        cout << files[i] << endl;
    }

    //Mat image;
    //image = imread( argv[1], 1 );


    namedWindow("Display Image", CV_WINDOW_AUTOSIZE );

    //imshow("Display Image", result);
    waitKey(0);


    return 0;
}