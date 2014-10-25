/* Deniz Sökmen S003244 Department of Computer Science */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>



using namespace cv;



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