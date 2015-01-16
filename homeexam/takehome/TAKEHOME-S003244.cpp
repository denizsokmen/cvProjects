#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <dirent.h>
#include <iostream>
#include <map>
#include <vector>
#include <math.h>



using namespace cv;
using namespace std;


#define PI 3.14159265

map<string, Mat> imageMap= map<string, Mat>();
map<double, vector<double>> cluster = map <double, vector<double>>();
map<double, double> clusterLengths = map<double, double>();
vector<double> results = vector<double>();

RNG rng(12345);

void help()
{
    cout << "\nThis program demonstrates line finding with the Hough transform.\n"
	"Usage:\n"
	"./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

void normalize_point(Vec2d &point) {
    double norm = cv::norm(point);
    point[0] /= norm;
    point[1] /= norm;
}

void addCluster(double orientation, double length) {
    if (cluster.empty()) {
	cluster[orientation].push_back(orientation);

	if (length > clusterLengths[orientation]) {
	    clusterLengths[orientation] = length;
	}

	return;
    }
    double threshold = 20 * PI / 180.0;
    bool found = false;
    for(auto it = cluster.begin(); it != cluster.end(); ++it) {
	if (abs(it->first - orientation) < threshold) {
	    cluster[it->first].push_back(orientation);
	    if (length > clusterLengths[it->first]) {
		clusterLengths[it->first] = length;
	    }
	    found = true;
	    break;
	}
    }
    
    if (!found) {
	cluster[orientation].push_back(orientation);
	if (length > clusterLengths[orientation]) {
	    clusterLengths[orientation] = length;
	}
    }
}

int pointToLineDist(Vec2d begin, Vec2d end, Vec2d point) {
    Vec2d AC = point - begin;
    Vec2d unitVec = end - begin;
    //cv::normalize(unitVec);
    normalize_point(unitVec);
    // unitVec = cv::normalize(unitVec);
    double norm = cv::norm(end-begin);
    //fprintf(stderr, "(%f, %f)\n", unitVec[0], unitVec[1]);
    double res = unitVec.dot(AC);
    
    // fprintf(stderr, "%f\n", res);
    if (res < 0) return cv::norm(point - begin);

    if (res > norm) return cv::norm(point - end);
    
    unitVec *= norm;
    
    Vec2d pointOnLine = begin + unitVec;
    
    return cv::norm(point - pointOnLine);
}

int traverseImages(string dir)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL) {
	return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
	if (dirp->d_type & DT_DIR) {
	    char path[1024];
	    int len = snprintf(path, sizeof(path)-1, "%s/%s", dir.c_str(), dirp->d_name);
	    path[len] = 0;
	    if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
		continue;

	    traverseImages(path);
	}
	else {

	    int len = strlen(dirp->d_name);

	    if (len >= 4) {
		char path[1024];
		int len = snprintf(path, sizeof(path)-1, "%s/%s", dir.c_str(), dirp->d_name);
		Mat histogram = imread(path, 0);

		imageMap[dirp->d_name] = histogram;
//		if (histogram.size().width < 400) {
		    Size size(500, 500);
		    Mat dest;
		    // erode(histogram, histogram, Mat(), Point(-1, -1), 1, 1, 1);
		    resize(histogram, dest, size);
		    imageMap[dirp->d_name] = dest;
		    //	}
	    }

	    // fprintf(stderr, "%s\n", dirp->d_name);

	}
    }
    closedir(dp);
    return 0;
}


// Ending point is further
void swapLineEnding(Vec4i &line,  Vec2i center) {
    Vec2i begin(line[0], line[1]);
    Vec2i end(line[2], line[3]);
    if (norm(begin, center) > norm(end, center)) {
	line[0] = end[0];
	line[1] = end[1];
	line[2] = begin[0];
	line[3] = begin[1];
    }
    else {
	line[2] = end[0];
	line[3] = end[1];
	line[0] = begin[0];
	line[1] = begin[1];
    }
    
    
}



void detectClock(string title, Mat image) {
    Mat dst, cdst;
     bitwise_not(image, image);
    //  cv::threshold(image, cdst, 0, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU);
    //dilate(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
     //  dilate(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
    // erode(image, dst, Mat(), Point(-1, -1), 1, 1, 1);
      Canny(image, dst, 50, 200, 3);
  
       cvtColor(dst, cdst, CV_GRAY2BGR);
    
  
     imshow(title, cdst);

}




int main(int argc, char** argv)
{

    traverseImages("./test");
    
    for (auto it = imageMap.begin(); it != imageMap.end(); ++it) {
	detectClock(it->first, it->second);
    }

    //fprintf(stderr, "%f\n", atan2(1, 0));


    waitKey();

    return 0;
}










