/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>



using namespace cv;


Mat applyFilter(const Mat& image, const Mat& filter) {
    Mat dest(image.rows - 2*(filter.rows / 2), image.cols - 2*(filter.cols / 2), CV_8UC3, cvScalar(0));

    // Temporary buffer to hold negative or big numbers.
    Vec3i *avg = new Vec3i[dest.rows*dest.cols];

    // Scaling for negatives or above-255 numbers
    int maxVal = 255;
    int minVal = 0;

    for (int row = filter.rows / 2, destRow = 0; row < image.rows - filter.rows / 2; row++, destRow++) {
        for (int col = filter.cols / 2, destCol = 0; col < image.cols - filter.cols / 2; col++, destCol++) {
            avg[dest.cols * destRow + destCol].val[0] = 0;
            avg[dest.cols * destRow + destCol].val[1] = 0;
            avg[dest.cols * destRow + destCol].val[2] = 0;

            // Summation of multiplication of filter and convoluted part of image
            Vec3i sum(0,0,0);
            for (int filterRow = 0; filterRow < filter.rows; filterRow++) {
                for (int filterCol = 0; filterCol < filter.cols; filterCol++) {
                    int offsetRow = row + filterRow - filter.rows / 2;
                    int offsetCol = col + filterCol - filter.cols / 2;
                    float color = filter.at<float>(filterRow, filterCol);
                    Vec3i imageCol = image.at<Vec3b>(offsetRow, offsetCol);


                    sum.val[0] += imageCol.val[0] * color;
                    sum.val[1] += imageCol.val[1] * color;
                    sum.val[2] += imageCol.val[2] * color;


                    // Precalculate the max and min values for scaling operations.
                    maxVal = max(maxVal, max(sum.val[0], max(sum.val[1], sum.val[2])));
                    minVal = min(minVal, min(sum.val[0], min(sum.val[1], sum.val[2])));
                }
            }

            avg[dest.cols * destRow + destCol] = sum;
        }
    }

    float dist = maxVal - minVal;

    // Reprocess the image to scale values and put on the destination.
    for (int row = 0; row < dest.rows; row++) {
        for (int col = 0; col < dest.cols; col++) {
            Vec3b sum;
            sum.val[0] = 255.0f* (avg[dest.cols * row + col].val[0] - minVal) / (dist);
            sum.val[1] = 255.0f* (avg[dest.cols * row + col].val[1] - minVal) / (dist);
            sum.val[2] = 255.0f* (avg[dest.cols * row + col].val[2] - minVal) / (dist);
            dest.at<Vec3b>(row, col) = sum;
        }
    }

    delete[] avg;

    return dest;
}
 
int main(int argc, char** argv )
{
    if (argc != 4) {
        fprintf(stderr, "Usage: [inputfilename] [filter.csv] [outputfilename]\n");
        return 0;
    }

    Mat image;
    image = imread( argv[1], 1 );

    // Read the CSV
    CvMLData mlData;
    mlData.read_csv(argv[2]);
    Mat filter(mlData.get_values(), true);
    
    namedWindow("Display Image", CV_WINDOW_AUTOSIZE );

    if ( !image.data )
    {
        // Not found or not an image, maybe a video?

        VideoCapture video(argv[1]);

        if (video.isOpened()) {
            fprintf(stderr, "Warning: If you haven\'t compiled OpenCV with a codec, video functionality may not work.\n");

            // Create a video writer using MPEG codec.
            Size size = Size(video.get(CV_CAP_PROP_FRAME_WIDTH),video.get(CV_CAP_PROP_FRAME_HEIGHT));
            VideoWriter writer(argv[3], CV_FOURCC('M','P','E','G'), video.get(CV_CAP_PROP_FPS), size);


            // Read the video frame by frame and write processed one
            while(true) {
                Mat frame;
                if (video.read(frame)) {
                    Mat res = applyFilter(frame, filter);
                    imshow("Display Image", res);
                    waitKey(1);
                    writer.write(res);
                }
                else {
                    break;
                }
            }
        }
        else {
            fprintf(stderr, "No image data or video \n");
            return -1;
        }
    }
    else {
        // Image found, filter it and write.
        Mat result = applyFilter(image, filter);
        imshow("Display Image", result);
        imwrite(argv[3], result);
        waitKey(0);
    }

    return 0;
}