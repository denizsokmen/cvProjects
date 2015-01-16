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

struct mapPointComparator {
    bool operator()(const Rect& a, const Rect& b) const {
        return a.x < b.x || a.y < b.y;
    }
};

Mat dice, canny;
int clusters = 0;
map<string, Mat> imageMap= map<string, Mat>();
map<int, vector<Point>> cluster = map <int, vector<Point>>();
map<Rect, vector<Point>, mapPointComparator> squareCluster = map<Rect,vector<Point>, mapPointComparator>();
vector<Point> circles = vector<Point>();
vector<Rect> squares = vector<Rect>();


void addCluster(Point orientation, double threshold) {
    if (cluster.empty()) {
        cluster[clusters++].push_back(orientation);
        
        
        return;
    }
    
    bool found = false;
    for(auto it = cluster.begin(); it != cluster.end(); ++it) {
        
        int xsum = 0;
        int ysum = 0;
        for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
            xsum += it2->x;
            ysum += it2->y;
        }
        Point p = Point(xsum / it->second.size(), ysum / it->second.size());
        if (cv::norm((p) - orientation) < threshold/2) {
            cluster[it->first].push_back(orientation);
            found = true;
            break;
        }
        
    }
    
    if (!found) {
        cluster[clusters++].push_back(orientation);
    }
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
            
            if (strncmp(dirp->d_name, ".", 1) == 0)
                continue;
            
            if (len >= 4) {
                char path[1024];
                int len = snprintf(path, sizeof(path)-1, "%s/%s", dir.c_str(), dirp->d_name);
                Mat histogram = imread(path, 0);
                
                imageMap[dirp->d_name] = histogram;
                Size size(500, 500);
                Mat dest;
                resize(histogram, dest, size);
                imageMap[dirp->d_name] = dest;
            }
            
            
        }
    }
    closedir(dp);
    return 0;
}


void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Rect r = cv::boundingRect(contour);
    cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}




void detectClock(string title, Mat image) {
    dice = image;
    cluster.clear();
    squareCluster.clear();
    squares.clear();
    circles.clear();
    clusters = 0;
    
    cv::Mat bw;
    cv::threshold(dice, bw, 0, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU);
    cv::Canny(bw, bw, 70, 220);
    dilate(bw, bw, Mat(), Point(-1, -1), 1, 1, 1);
    // Find contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(bw.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> approx;
    cv::Mat dst = dice.clone();
    for (int i = 0; i < contours.size(); i++)
    {
        
        cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
        
        if (std::fabs(cv::contourArea(contours[i])) < 10 || !cv::isContourConvex(approx))
            continue;
        if (approx.size() >= 4 && approx.size() <= 6)
        {
            // Number of vertices of polygonal curve
            int vtc = approx.size();
            // Get the cosines of all corners
            std::vector<double> cos;
            for (int j = 2; j < vtc+1; j++)
                cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));
            // Sort ascending the cosine values
            std::sort(cos.begin(), cos.end());
            // Get the lowest and the highest cosine
            double mincos = cos.front();
            double maxcos = cos.back();
            // Use the degrees obtained above and the number of vertices
            // to determine the shape of the contour
            if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3) {
                setLabel(dst, "RECT", contours[i]);
                cv::Rect r = cv::boundingRect(contours[i]);
                cv::Point pt(r.x + ((r.width) / 2), r.y + ((r.height) / 2));
                bool found = false;
                
                for (auto it = squares.begin(); it != squares.end(); it++) {
                    if (norm(Point(it->x, it->y) - Point(r.x, r.y)) < 10)
                        found = true;
                }
                
                if (!found)
                    squares.push_back(r);
            }
        }
        else
        {
            setLabel(dst, "CIR", contours[i]);
            cv::Rect r = cv::boundingRect(contours[i]);
            cv::Point pt(r.x + ((r.width) / 2), r.y + ((r.height) / 2));
            bool found = false;
            
            for (auto it = circles.begin(); it != circles.end(); it++) {
                if (norm(*it - pt) < 5)
                    found = true;
            }
            
            if (!found)
            circles.push_back(pt);
        }
    }

    
    
    for(auto circ = circles.begin(); circ != circles.end(); circ++) {
        
        Point nearest;
        double len = 1000000.0;
        for (auto sq = squares.begin(); sq != squares.end(); sq++) {
            rectangle(dst, Point(sq->x, sq->y), Point(sq->x + sq->width, sq->y + sq->height), Scalar(0));
        
            if (circ->x >= sq->x && circ->x <= sq->x + sq->width && circ->y >= sq->y && circ->y <= sq->y + sq->height ) {
                squareCluster[*sq].push_back(*circ);
                
                line(dst, *circ, Point(sq->x + ((sq->width) / 2), sq->y + ((sq->height) / 2)), Scalar(0,0,0));
                break;
            }
            
        }
    }
    
    int hist[6] = {0, 0, 0, 0, 0, 0};
    
    for(auto it = squareCluster.begin(); it != squareCluster.end(); it++) {
        for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
            circles.erase(remove(circles.begin(), circles.end(), *it2), circles.end());
        }
    }
    
    
    //lambda rulez
    sort(circles.begin(), circles.end(),
         [](const Point & a, const Point & b) -> bool
    {
        return norm(a) < norm(b);
    });
    
    
    Rect r;
    
    for(auto it = squares.begin(); it != squares.end(); it++) {
        if (squareCluster[*it].size() > 0) {
            r = *it;
            break;
        }
    }
   
    for (auto it = circles.begin(); it != circles.end(); it++) {
        addCluster(*it, norm(Point(r.width, r.height)));
    }
    
    for(auto it = cluster.begin(); it != cluster.end(); it++) {
        
        Rect newrect;
        
        int xsum = 0;
        int ysum = 0;
        for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
            xsum += it2->x;
            ysum += it2->y;
        }
        
        newrect.x = xsum/it->second.size() - r.width/2;
        newrect.y = ysum/it->second.size() - r.height/2;
        newrect.width = r.width;
        newrect.height = r.height;
        
        rectangle(dst, Point(newrect.x, newrect.y), Point(newrect.x + newrect.width, newrect.y + newrect.height), Scalar(0));
        
        squares.push_back(newrect);
        
    }
    
    for(auto circ = circles.begin(); circ != circles.end(); circ++) {
        
        Point nearest;
        double len = 1000000.0;
        for (auto sq = squares.begin(); sq != squares.end(); sq++) {
            rectangle(dst, Point(sq->x, sq->y), Point(sq->x + sq->width, sq->y + sq->height), Scalar(0));
            if (circ->x >= sq->x && circ->x <= sq->x + sq->width && circ->y >= sq->y && circ->y <= sq->y + sq->height ) {
                squareCluster[*sq].push_back(*circ);
                
                line(dst, *circ, Point(sq->x + ((sq->width) / 2), sq->y + ((sq->height) / 2)), Scalar(0,0,0));
                break;
            }
            
        }
    }
    
    for(auto it = squareCluster.begin(); it != squareCluster.end(); it++) {
        int toput = it->second.size() - 1;
        if (toput > 5)
            toput = 5;
        hist[toput]++;
        
    }
    
    
    fprintf(stderr, "%s : [%d, %d, %d, %d, %d, %d]\n", title.c_str(), hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
    namedWindow(title.c_str());
    imshow(title.c_str(), dst);
    waitKey();
    
}




int main(int argc, char** argv)
{
    
    traverseImages("./test");
    
    for (auto it = imageMap.begin(); it != imageMap.end(); ++it) {
        detectClock(it->first, it->second);
    }
    
    waitKey();
    
    return 0;
}










