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

map<string, Mat> imageMap= map<string, Mat>();
map<double, vector<double>> cluster = map <double, vector<double>>();

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

void addCluster(double orientation) {
    if (cluster.empty()) {
	cluster[orientation].push_back(orientation);
	return;
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
    
    fprintf(stderr, "%f\n", res);
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
	    }

	    fprintf(stderr, "%s\n", dirp->d_name);

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
    cv::threshold(image, dst, 0, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU);
    // dilate(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
    erode(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
    //Canny(dst, dst, 50, 200, 3);
  
    cvtColor(dst, cdst, CV_GRAY2BGR);

    
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 10, 30, 5 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
	Vec4i l = lines[i];
	swapLineEnding(l, Vec2i(cdst.cols / 2, cdst.rows / 2));
//	fprintf(stderr, "%d %d %d %d\n", l[0], l[1], l[2], l[3]);
	if (pointToLineDist(Vec2d(l[0], l[1]), Vec2d(l[2], l[3]), Vec2d(cdst.cols / 2, cdst.rows / 2)) < 80)
	    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }

    //   kmeans(InputArray data, int K, InputOutputArray bestLabels, cv::TermCriteria criteria, int attempts, int flags
     imshow(title, cdst);

}




int main(int argc, char** argv)
{

    traverseImages("./dataset");
    
    for (auto it = imageMap.begin(); it != imageMap.end(); ++it) {
	detectClock(it->first, it->second);
    }


    waitKey();

    return 0;
}










