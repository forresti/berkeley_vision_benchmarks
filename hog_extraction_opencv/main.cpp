
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;
using namespace cv;

vector<float> wrappedCvHog(cv::Mat img)
{
    //magic parameters -- see experienceopencv.blogspot.com/2011/02/hog-descriptor.html
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    cv::Size block_size(16,16);
    cv::Size block_stride(8,8);
    cv::Size cell_size(8,8);
    int nOri = 9;

    //off-the-shelf version was from opencv/opencv/samples/ocl/hog.cpp (search for cpu_hog) 
    cv::HOGDescriptor d(win_size, block_size, block_stride, cell_size, nOri, 1, -1,
                              cv::HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS); //configure HOG extractor

    vector<float> features;
    vector<cv::Point> locations;
    d.compute(img, features, cv::Size(0,0), cv::Size(0,0), locations); //run HOG extractor
    printf("features.size() = %d \n", (int)features.size());
    return features;
}

//based on samples/cpp/peopledetect.cpp in opencv distribution
vector<Rect> wrappedCvHogPedestrian(cv::Mat img)
{
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> detections;
    hog.detectMultiScale(img, detections, 0, Size(8,8), Size(32,32), 1.05, 2); //magic numbers from peopledetect.cpp
}

void benchmarkOpenCvHOG()
{
    Mat img = imread("Lena.pgm");
    cv::cvtColor(img, img, CV_RGB2GRAY);   

  //benchmark HOG extraction only (no detection)
    double start = read_timer();
    vector<float> features = wrappedCvHog(img); 
    double responseTime = read_timer() - start;
    printf("OpenCV HOG time = %f \n", responseTime);

    Mat hog_visualization = get_hogdescriptor_visu(img, features);   //off-the-shelf code that uses OpenCV; see helpers.cpp
    forrestWritePgm(hog_visualization, "Lena_hog.pgm"); //uses OpenCV

  //benchmark HOG extraction + pedestrian detection
    start = read_timer();
    vector<Rect> detections = wrappedCvHogPedestrian(img);
    responseTime = read_timer() - start;
    printf("OpenCV HOG + pedestrian detection time = %f \n", responseTime);

    Mat detection_visualization = drawDetections(img, detections);
}

int main (int argc, char **argv)
{
    benchmarkOpenCvHOG();    

    return 0;
}
