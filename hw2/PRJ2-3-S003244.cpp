/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>


using namespace cv;
using namespace std;

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct stat statbuff;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
            continue;
        stat(dirp->d_name, &statbuff);

        //if (S_ISDIR(statbuff.st_mode))
          //  continue;
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

int main(int argc, char** argv )
{
    string dir = string(".");
    vector<string> files = vector<string>();

    getdir(dir,files);

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