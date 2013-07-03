
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;
using namespace cv;


//run OpenCV filter2D on a real image
double demoOpenCvConvolution()
{
    Mat img = imread("./Lena.pgm");
    Mat outImg = img.clone(); //allocate space for the convolution results 

    Mat kernel = Mat::zeros(3, 3, CV_32FC1);
    kernel.at<float>(0,0)=-1;
    kernel.at<float>(0,1)= 0;
    kernel.at<float>(0,1)= 1;
    kernel.at<float>(1,0)=-2;
    kernel.at<float>(1,1)= 0;
    kernel.at<float>(1,2)= 2;
    kernel.at<float>(2,0)=-1;
    kernel.at<float>(2,1)= 0;
    kernel.at<float>(2,2)= 1;
    
    Point anchor(-1,-1);
    double delta = 0;
    int ddepth = -1; 
    
    double start = read_timer();
    filter2D(img, outImg, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
    double execTime = read_timer() - start;
    forrestWritePgm(outImg, "Lena_opencvFilter.pgm");     
    return execTime;
}

//returns computation time of OpenCV filter2D
double runOpenCvConvolution(int imgRows, int imgCols, int kernelRows, int kernelCols, string dataType)
{
    Mat img;
    if(dataType == "8u")
        img = Mat::zeros(imgRows, imgCols, CV_8UC1);
    if(dataType == "32f")
        img = Mat::zeros(imgRows, imgCols, CV_32FC1);
    Mat outImg = img.clone(); //allocate space for the convolution results 

    Mat kernel = Mat::ones(kernelRows, kernelCols, CV_32FC1);
    Point anchor(-1,-1);
    double delta = 0;
    int ddepth = -1; 
    
    double start = read_timer();
    filter2D(img, outImg, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
    double execTime = read_timer() - start;
    return execTime;
}

//returns computation time of OpenCV GPU filter2D
double runOpenCvGPUConvolution(int imgRows, int imgCols, int kernelRows, int kernelCols, string dataType)
{
    Mat img;
    if(dataType == "8u")
        img = Mat::zeros(imgRows, imgCols, CV_8UC1);
    if(dataType == "32f")
        img = Mat::zeros(imgRows, imgCols, CV_32FC1);
    Mat outImg = img.clone(); //allocate space for the convolution results 

    gpu::GpuMat gImg;
    gpu::GpuMat gOutImg;
    gImg.upload(img);
    gOutImg.upload(outImg); //make sure space is preallocated

    Mat kernel = Mat::ones(kernelRows, kernelCols, CV_32FC1);

    Point anchor(-1,-1);
    double delta = 0;
    int ddepth = -1;

    double start = read_timer();
    gpu::filter2D(gImg, gOutImg, ddepth, kernel, anchor, BORDER_DEFAULT); //note that kernel is a host Mat 

    double execTime = read_timer() - start;
    return execTime;
}

void benchmarkOpenCvConvolution()
{
    int imgRows = 9000;
    int imgCols = 9000;
    string dataType = "32f";
    int nRuns = 10;
    for(int kernelSize = 2; kernelSize < 9; kernelSize++)
    {
        double execTime = 0;
        for(int i=0; i<nRuns; i++)
        {
            //execTime += runOpenCvConvolution(imgRows, imgCols, kernelSize, kernelSize, dataType);
            execTime += runOpenCvGPUConvolution(imgRows, imgCols, kernelSize, kernelSize, dataType);
        }
        printf("imgSize = %dx%d,  kernelSize = %d, avg execTime = %f \n", imgCols, imgRows, kernelSize, execTime/nRuns);
    }
#if 0
    kernelSize = 5;
    for(int imgSize=256; imgSize<20000; imgSize*=2) //imgSize = imgRows = imgCols
    {
        double execTime = runOpenCvConvolution(imgSize, imgSize, kernelSize, kernelSize, dataType);
        //double execTime = runOpenCvGPUConvolution(imgSize, imgSize, kernelSize, kernelSize, dataType);
        printf("imgSize = %dx%d,  kernelSize = %d,  execTime = %f \n", imgSize, imgSize, kernelSize, execTime);
    }
#endif
}

vector<float> wrappedCvHog(cv::Mat img)
{
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    //int c = 4;
    //cv::Size block_size(c,c);
    //cv::Size block_stride(c,c); //I think this is to select whether to do overlapping blocks
    //cv::Size cell_size(c,c);
    cv::Size block_size(16,16);
    cv::Size block_stride(8,8);
    cv::Size cell_size(8,8);
    int nOri = 15;

    //off-the-shelf version was from opencv/opencv/samples/ocl/hog.cpp (search for cpu_hog) 
    cv::HOGDescriptor d(win_size, block_size, block_stride, cell_size, nOri, 1, -1,
                              cv::HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);

    vector<float> features;
    vector<cv::Point> locations;
    d.compute(img, features, cv::Size(0,0), cv::Size(0,0), locations);
    printf("features.size() = %d \n", (int)features.size());
    return features;
}

void wrappedCvHogGPU(cv::Mat img)
{
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    //int c = 4;
    //cv::Size block_size(c,c);
    //cv::Size block_stride(c,c); //I think this is to select whether to do overlapping blocks
    //cv::Size cell_size(c,c);
    cv::Size block_size(16,16);
    cv::Size block_stride(8,8);
    cv::Size cell_size(8,8);
    int nOri = 15;

    cv::gpu::HOGDescriptor d(win_size, block_size, block_stride, cell_size, nOri, cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA,
                              cv::HOGDescriptor::L2Hys, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
    //cv::gpu::HOGDescriptor d; //there are assert statements that pretty much require default configuration
    
    gpu::GpuMat dImg;
    dImg.upload(img); //this hangs, even with a small image.
    gpu::GpuMat features;
    cv::Size win_stride(img.rows, img.cols); //no stride, I guess
    vector<Point> found_locations;

    //d.detect(dImg, found_locations);

    double start = read_timer();
    d.getDescriptors(dImg, win_stride, features);
    double responseTime = read_timer() - start;
    printf("GPU HOG getDescriptors() time = %f \n", responseTime);

    printf("features size = %d \n", features.rows * features.cols);
}

void benchmarkOpenCvHOG()
{
    Mat img = imread("../forrest_hacked_OpenCV_2_Cookbook_Code/9k_x_9k.png");
    //Mat img = imread("../forrest_hacked_OpenCV_2_Cookbook_Code/Lena_orig.png");
    cv::cvtColor(img, img, CV_RGB2GRAY);   

    double start = read_timer();
    wrappedCvHog(img); 
    double responseTime = read_timer() - start;
    printf("CPU OpenCV HOG time = %f \n", responseTime);

    start = read_timer();
    wrappedCvHogGPU(img); 
    responseTime = read_timer() - start;
    printf("GPU OpenCV HOG time (including memcpy) = %f \n", responseTime);
}

int main (int argc, char **argv)
{
    //cudaSetDevice(3); //C2050
    //demoOpenCvConvolution();
    benchmarkOpenCvConvolution();
    //benchmarkOpenCvHOG();    

    return 0;
}
