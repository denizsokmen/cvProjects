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

int p1, p2;
Mat dice, canny;
int clusters = 0;
map<string, Mat> imageMap= map<string, Mat>();
map<int, vector<Point>> cluster = map <int, vector<Point>>();
map<double, double> clusterLengths = map<double, double>();
vector<double> results = vector<double>();

RNG rng(12345);

void help()
{
    cout << "\nThis program demonstrates line finding with the Hough transform.\n"
    "Usage:\n"
    "./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}


void addCluster(Point orientation) {
    if (cluster.empty()) {
        cluster[clusters++].push_back(orientation);
        
        
        return;
    }
    
    bool found = false;
    for(auto it = cluster.begin(); it != cluster.end(); ++it) {
        for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            if (cv::norm((*it2) - orientation) < 50) {
                cluster[it->first].push_back(orientation);
                found = true;
                break;
            }
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




void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();
    
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    
    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);
    
    int label_count = 2; // starts at 2 because 0,1 are used already
    
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }
            
            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
            
            std::vector <cv::Point2i> blob;
            
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    
                    blob.push_back(cv::Point2i(j,i));
                }
            }
            
            blobs.push_back(blob);
            
            label_count++;
        }
    }
}

void on_trackbar(int, void*) {
    Canny(dice, canny, p1, p2);
    dilate(canny, canny, Mat(), Point(-1, -1), 1, 1, 1);
    /*dilate(canny, canny, Mat(), Point(-1, -1), 1, 1, 1);
     dilate(canny, canny, Mat(), Point(-1, -1), 1, 1, 1);
     dilate(canny, canny, Mat(), Point(-1, -1), 1, 1, 1);
     dilate(canny, canny, Mat(), Point(-1, -1), 1, 1, 1);*/
    imshow("canny", canny);
}

void detectClock(string title, Mat image) {
    /*Mat dst, cdst;
     bitwise_not(image, image);
     cv::threshold(image, cdst, 0, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU);
     //dilate(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
     //  dilate(dst, dst, Mat(), Point(-1, -1), 1, 1, 1);
     
     Canny(image, image, 60, 150, 3);
     
     
     //dilate(image, image, Mat(), Point(-1, -1), 1, 1, 1);
     
     cvtColor(image, cdst, CV_GRAY2BGR);
     
     cv::Mat output = cv::Mat::zeros(image.size(), CV_8UC3);
     
     cv::Mat binary;
     std::vector < std::vector<cv::Point2i > > blobs;
     
     cv::threshold(image, binary, 0.0, 1.0, cv::THRESH_BINARY);
     
     FindBlobs(binary, blobs);
     
     // Randomy color the blobs
     for(size_t i=0; i < blobs.size(); i++) {
     unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
     unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
     unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));
     
     for(size_t j=0; j < blobs[i].size(); j++) {
     int x = blobs[i][j].x;
     int y = blobs[i][j].y;
     
     output.at<cv::Vec3b>(y,x)[0] = b;
     output.at<cv::Vec3b>(y,x)[1] = g;
     output.at<cv::Vec3b>(y,x)[2] = r;
     }
     }
     
     cv::imshow("binary", image);
     cv::imshow("labelled", output);
     cv::waitKey(0);
     
     imshow(title, cdst);*/
    dice = image;
    p1 = 100;
    p2 = 200;
    Canny(dice, canny, 375, 820);
    imshow("canny", canny);
    createTrackbar("p1","canny",&p1,1000,on_trackbar);
    createTrackbar("p2","canny",&p2,1000,on_trackbar);
    waitKey();
    int num = 0;
    for(int y=0;y<canny.size().height;y++)
    {
        uchar *row = canny.ptr(y);
        for(int x=0;x<canny.size().width;x++)
        {
            if(row[x] <= 128)
            {
                int area = floodFill(canny, Point(x,y), CV_RGB(0,0,160));
                printf("filling %d, %d gray, area is %d\n", x, y, area);
                if(area>10 && area < 250) {
                    addCluster(Point(x,y));
                    num++;
                }
            }
        }
    }
    
    for(auto it = cluster.begin(); it != cluster.end(); it++) {
        fprintf(stderr, "%lu\n", it->second.size());
    }
    printf("number is %d\n", num);
    imshow("dice", canny);
    waitKey();
    
    
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










