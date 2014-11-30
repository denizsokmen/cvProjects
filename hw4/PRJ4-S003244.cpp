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
    //cv::threshold(image, dst, 0, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU);
    // dilate(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
    // dilate(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
    erode(image, dst, Mat(), Point(-1, -1), 1, 1, 1);
    Canny(dst, dst, 50, 200, 3);
  
    cvtColor(dst, cdst, CV_GRAY2BGR);

    fprintf(stderr, "%s", title.c_str());
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 1 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
	Vec4i l = lines[i];
	swapLineEnding(l, Vec2i(cdst.cols / 2, cdst.rows / 2));
	Vec2i diff = Vec2i(l[2],l[3]) - Vec2i(l[0],l[1]);
	Vec2i center(cdst.cols / 2, cdst.rows / 2);
//	fprintf(stderr, "%d %d %d %d\n", l[0], l[1], l[2], l[3]);
	if (pointToLineDist(Vec2d(l[0], l[1]), Vec2d(l[2], l[3]), Vec2d(cdst.cols / 2, cdst.rows / 2)) < 90) {
          
	    addCluster(atan2(-diff[1], diff[0]), norm(Vec2i(l[2], l[3])-center));

	    // fprintf(stderr, "(%d,%d) (%d, %d) (%d, %d) %f %f\n", l[0], l[1], l[2], l[3], cdst.cols / 2, cdst.rows / 2, norm(Vec2i(l[0], l[1]) - center), norm(Vec2i(l[2], l[3]) - center));
	    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
	    line( cdst, Point(center[0], center[1]), Point(center[0]+2, center[1]+2), Scalar(0,255,0), 3, CV_AA);
	}
    }

    for (auto it = cluster.begin(); it != cluster.end(); ++it) {
//	fprintf(stderr, "(%f)\n", it->first);

	double sum = 0.0;
	for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
	    // fprintf(stderr, "--(%f)\n", *it2);
	    sum += *it2;
	}

	sum /= it->second.size();
	results.push_back(sum);
    }

    int hour = 0;
    int minute = 0;
    
    auto cmp = [](std::pair<double,double> const & a, std::pair<double,double> const & b) 
	{ 
	    return a.second != b.second?  a.second < b.second : a.first < b.first;
	};
    vector<pair<double,double>> hands(clusterLengths.begin(), clusterLengths.end());
    sort(hands.begin(), hands.end(), cmp);

    /*for (auto it = hands.begin(); it != hands.end(); it++) {
	fprintf(stderr, "%f ... %f\n", it->first, it->second);
	}*/
    
    if (!hands.empty()) {
	double degrees = 360 - (fmod((hands.front().first + PI*2), (PI*2)) * 180.0 / PI);
	double degrees2 = 360 - (fmod((hands.back().first + PI*2), (PI*2)) * 180.0 / PI);

//	fprintf(stderr, "%f ... %f\n", degrees, degrees2);
	hour = ((int)(3 + (degrees / 30))) % 12;
	minute = ((int)(15 + (degrees2 / 6))) % 60;
    }

    fprintf(stderr, " = %02d:%02d or ", hour, minute);

    if (!hands.empty()) {
	double degrees = 360 - (fmod((hands.front().first + PI*2), (PI*2)) * 180.0 / PI);
	double degrees2 = 360 - (fmod((hands.back().first + PI*2), (PI*2)) * 180.0 / PI);

//	fprintf(stderr, "%f ... %f\n", degrees, degrees2);
	hour = ((int)(3 + (degrees2 / 30))) % 12;
	minute = ((int)(15 + (degrees / 6))) % 60;
    }

    fprintf(stderr, "%02d:%02d\n", hour, minute);

    clusterLengths.clear();
    cluster.clear();
    results.clear();
    imshow(title, cdst);

}




int main(int argc, char** argv)
{

    traverseImages("./dataset");
    
    for (auto it = imageMap.begin(); it != imageMap.end(); ++it) {
	detectClock(it->first, it->second);
    }

    //fprintf(stderr, "%f\n", atan2(1, 0));


    waitKey();

    return 0;
}










