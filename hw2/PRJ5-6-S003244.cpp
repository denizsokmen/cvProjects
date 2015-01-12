/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>

#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <thread>
#include <fstream>


#define QUANTIZATION 8
#define GRID_SIZE 4 // 4 x 4

using namespace cv;
using namespace std;


map<string, vector<Mat>> trainingSet = map<string, vector<Mat>>();
map<string, vector<Mat>> testSet = map<string, vector<Mat>>();

map<string, vector<Mat>> projectedTrainingSet = map<string, vector<Mat>>();
map<string, vector<Mat>> projectedTestSet = map<string, vector<Mat>>();

bool is_valid_double(double x)
{
    return x*0.0==0.0;
}

Mat getHistogram(Mat& image, int quantum, int grid) {
    int numBins = 256/quantum;
    int totalBins = image.channels() * numBins;
    int gridSquare = grid*grid;

    Mat dest(1, totalBins * gridSquare, CV_64FC1, cvScalar(0));

    map<int, double> bins = map<int, double>();

    double sumofall = 0.0;
    for (int i = 0; i < dest.cols; i++) {
        bins[i] = 0.0f;
    }

    for (int i = 0; i < image.rows; i++) {
	for (int j = 0; j < image.cols; j++) {
	    Vec3b col = image.at<Vec3b>(i, j);
	    int offsetRow = min(grid-1, (i-1) / (image.rows / grid));
	    int offsetCol = min(grid-1, (j-1) / (image.cols / grid));
	    int offset = offsetCol * grid + offsetRow;

	    for (int k = 0; k < image.channels(); k++) {
		int illumination = col.val[k];
		int bin = (totalBins*offset) + (numBins * k) + (illumination / quantum);
		bins[bin] += 1.0;
		sumofall += 1.0;
	    }
	}
    }
    

    for(auto it = bins.begin(); it != bins.end(); ++it) {
        int offset = it->first / totalBins;

	if (is_valid_double(sumofall))
	    it->second /= sumofall;
	else {
	    fprintf(stderr, "%f %d\n", sumofall, offset);
	    exit(0);
	}
    }

    for (int i = 0; i < dest.cols; i++) {
        dest.at<double>(0, i) = bins[i];
    }


    //  subtract(dest, mean(dest), dest);
    
    // delete[] sumofall;

    return dest;
}


// I have taken this function from a blog. It converts vector of Mats into
// a single Mat to be used for PCA.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    fprintf(stderr, "%d %d\n",src.size(), d);
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
	src[i].copyTo(xi);

	/* if(src[i].isContinuous()) {
	     src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
	     src[i].clone().reshape(1, 1).convertTo(xi, alpha,beta);
	     }*/
    }
    return data;
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

int traverseImages(string dir, map<string, vector<Mat>> &trainingFiles, map<string, vector<Mat>> &testFiles, string category, int count)
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

            traverseImages(path, trainingFiles, testFiles, string(dirp->d_name), getNumOfJPG(path));

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
                    histogram = getHistogram(histogram, QUANTIZATION, GRID_SIZE);
		    
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

double intersectHist(Mat& h1, Mat& h2) {
    double sum = 0.0;
    
    for (int i = 0; i < h1.cols; i++) {
        double h1val = h1.at<float>(0, i);
        double h2val = h2.at<float>(0, i);
        //sum += min(h1val, h2val);
        if (h1val + h2val > 0.0)
            sum += (h1val - h2val) * (h1val-h2val) / (h1val + h2val);
        //sum += (h1.at<float>(0, i) - h2.at<float>(0, i)) * (h1.at<float>(0, i) - h2.at<float>(0, i)) / (h1.at<float>(0, i) + h2.at<float>(0, i));
    }
    //fprintf(stderr, "%f\n", sum);

    return sum/2.0;
}

void applyPCA(double variance) {
    projectedTestSet.clear();
    projectedTrainingSet.clear();
    fprintf(stderr, "Putting histograms as rows in a single matrix...");
    vector<Mat> histData;
    for (auto testIt = trainingSet.begin(); testIt != trainingSet.end(); ++testIt) {
        for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
	    Mat data = *testImg;
	    subtract(data, mean(data), data);
	    histData.push_back(data);
	}
    }

    Mat data = asRowMatrix(histData, CV_64FC1);
    fprintf(stderr, "Done.\n");


    fprintf(stderr, "Performing PCA with retained variance of %f...", variance);
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, variance);
    fprintf(stderr, "Done.\n");
     
    fprintf(stderr, "Projecting feature vectors on eigenspace...");

    for (auto testIt = testSet.begin(); testIt != testSet.end(); ++testIt) {
	for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
	    Mat proj = pca.project(*testImg);
	    projectedTestSet[testIt->first].push_back(proj);
	}
    }

    for (auto it = trainingSet.begin(); it != trainingSet.end(); ++it) {
	for (auto trainingImg = it->second.begin(); trainingImg != it->second.end(); trainingImg++) {
	    Mat proj = pca.project(*trainingImg);
	    projectedTrainingSet[it->first].push_back(proj);
	}
    }


    fprintf(stderr, "Done.\n");
}

void compareResults() {
    double minDist = 1000000.0;
    string minCategory = " ";
    map<string, vector<string>> results = map<string, vector<string>>();
    fprintf(stderr, "Comparing projected histograms...");
    // map the training category with test category
    for (auto testIt = projectedTestSet.begin(); testIt != projectedTestSet.end(); ++testIt) {
        for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
            for (auto it = projectedTrainingSet.begin(); it != projectedTrainingSet.end(); ++it) {
                for (auto trainingImg = it->second.begin(); trainingImg != it->second.end(); trainingImg++) {
                    double res = norm(*testImg, *trainingImg, NORM_L2);
		    // fprintf(stderr, "%f\n",res);
                    if (res < minDist) {
                        minDist = res;
                        minCategory = it->first;
                    }
                }
            }

            results[minCategory].push_back(testIt->first);
            minDist = 10000000.0;
        }
    }
    fprintf(stderr, "Done.\n");

    int matchedSize = 0;
    int testSize = 0;

    int totalSize = 0;
    int totalMatch = 0;
    

    fprintf(stderr, "Calculating results...\n");

    for (auto testIt = results.begin(); testIt != results.end(); ++testIt) {
        matchedSize = 0;
        testSize = 0;
        fprintf(stderr, "%s - ", testIt->first.c_str());
        testSize += projectedTestSet[testIt->first].size();
        totalSize += projectedTestSet[testIt->first].size();

        for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
            if (*testImg == testIt->first) {
                matchedSize++;
                totalMatch++;
            }

        }
        fprintf(stderr, "%f%\n", (float)100.0f*matchedSize / testSize);
    }

    fprintf(stderr, "Done\n");
    fprintf(stderr, "Overall: %f%\n", (float)100.0f*totalMatch / totalSize);
}

void writeResults(std::string filename) {
    ofstream myfile;
    myfile.open (filename);

    for (auto testIt = projectedTestSet.begin(); testIt != projectedTestSet.end(); ++testIt) {
	 for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
	     for (int i = 0; i < testImg->cols - 1; i++) {
		 myfile << testImg->at<double>(0,i) << ",";
	     }
	     myfile << testImg->at<double>(0, testImg->cols - 1) << "\n";
	 }
     }

     for (auto it = projectedTrainingSet.begin(); it != projectedTrainingSet.end(); ++it) {
	 for (auto trainingImg = it->second.begin(); trainingImg != it->second.end(); trainingImg++) {
	     for (int i = 0; i < trainingImg->cols - 1; i++) {
		 myfile << trainingImg->at<double>(0,i) << ",";
	     }
	     myfile << trainingImg->at<double>(0, trainingImg->cols - 1) << "\n";
	 }
     }

    myfile.close();
}

void addCluster(vector<Mat>& cluster, Mat weightedmean) {
    if (cluster.empty()) {
	cluster.push_back(weightedmean);
	return;
    }
    double threshold = 0.0003;
    bool found = false;
    for(auto it = cluster.begin(); it != cluster.end(); ++it) {
	double dist = norm(*it, weightedmean, NORM_L2);
//	fprintf(stderr, "dist: %f\n", dist);
	if (dist < threshold) {
	    found = true;
	    break;
	}
    }
    
    if (!found) {
	cluster.push_back(weightedmean);
    }

}

void meanShift(double bandwidth) {
    fprintf(stderr, "Started meanshift...\n");
    /*vector<Mat> dataset;
    

    for (auto testIt = projectedTestSet.begin(); testIt != projectedTestSet.end(); ++testIt) {
	for (auto testImg = testIt->second.begin(); testImg != testIt->second.end(); ++testImg) {
	    dataset.push_back(*testImg);
	}
    }

    for (auto it = projectedTrainingSet.begin(); it != projectedTrainingSet.end(); ++it) {
	for (auto trainingImg = it->second.begin(); trainingImg != it->second.end(); trainingImg++) {
	    dataset.push_back(*trainingImg);
	}
	}*/


    CvMLData mlData;
    mlData.read_csv("results70.csv");
    const CvMat* tmp = mlData.get_values();
    cv::Mat dataset(tmp, true);
    dataset.convertTo(dataset, CV_64FC1);

    Mat weights(1, dataset.rows, CV_64FC1, cvScalar(0));
    Mat distances(1, dataset.rows, CV_64FC1, cvScalar(0));
    Mat mult(1, dataset.rows, CV_64FC1, cvScalar(0));
    Mat oldweight(1, dataset.cols, CV_64FC1, cvScalar(0));
    Mat newweight;
    vector<Mat> clusters;
    for (int m = 0; m < dataset.rows; m++) {
	Mat point = dataset.row(m);
	newweight = point;
	

	do {
	    for (int n = 0; n < dataset.rows; n++) {
		Mat comp = dataset.row(n);
		double dist = norm(newweight, comp, NORM_L2);
	       	//fprintf(stderr, "%f,",dist);
		dist = -1.0 * dist * dist / (bandwidth * bandwidth);
		distances.at<double>(0, n) = dist;
	    }
	    //fprintf(stderr, "\n-----------\n");
	
	    exp(distances, weights);
	    Mat summation(1, point.cols, CV_64FC1, cvScalar(0));
	    for (int n = 0; n < weights.cols; n++) {
		double w = weights.at<double>(0,n);
		//	fprintf(stderr, "%f,",w);
		Mat comp = dataset.row(n);
		add(summation, w*comp, summation);		
	    }
	    //fprintf(stderr, "\n-----------\n");



	    double weightsum = sum(weights)[0];
	    oldweight = newweight.clone();
	    summation.copyTo(newweight);
	    newweight = newweight / weightsum;
	} while (norm(newweight, oldweight, NORM_L2) > 0.01);

	addCluster(clusters, newweight.clone());
    }


    
    /*map<double, vector<double>> cluster = map<double, vector<double>>();
    ofstream myfile;
    myfile.open ("meanshift.csv");

   

    for (auto it = clusters.begin(); it != clusters.end(); it++) {
        addCluster(cluster, norm(*it, NORM_L2));
	for (int j = 0; j < it->rows; j++) {
	    for (int i = 0; i < it->cols - 1; i++) {
		myfile << it->at<double>(j,i) << ",";
	    }
	    myfile << it->at<double>(j, it->cols - 1) << "\n";
	}
    }

    myfile.close();*/

    fprintf(stderr, "Clusters: %d \n", clusters.size());

    /*for (auto it = cluster.begin(); it != cluster.end(); it++) {
	fprintf(stderr, "[%f],", it->first);
    }
    fprintf(stderr, "\n");*/
    
    
}


int main(int argc, char** argv )
{
    string dir = string("./DATASET");
    fprintf(stderr, "Traversing images & calculating histograms...");
    traverseImages(dir, trainingSet, testSet, ".", 0);
    fprintf(stderr, "Done.\n");

    namedWindow("Display Image", CV_WINDOW_AUTOSIZE );

    
    
    applyPCA(0.70);
    compareResults();
    writeResults("results70.csv");
    
    meanShift(0.1);

    /* applyPCA(0.75);
    compareResults();
    writeResults("results75.csv");

    applyPCA(0.80);
    compareResults();
    writeResults("results80.csv");

    applyPCA(0.85);
    compareResults();
    writeResults("results85.csv");

    applyPCA(0.90);
    compareResults();
    writeResults("results90.csv");

    applyPCA(0.95);
    compareResults();
    writeResults("results95.csv");*/
    return 0;
}
