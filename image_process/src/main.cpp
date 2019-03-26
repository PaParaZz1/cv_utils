#include "png.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <stdio.h>

using std::string;

extern writePng(FILE* fp, const unsigned char* data, int w, int h);

void testWritePng() {
    FILE* fp;
    fp = fopen("test_result.png", "w");
    int h = 256;
    int w = 256;
    unsigned char data[h*w*3];
    unsigned char* ptr = data;
    for (int y=0; y<h; ++y) {
        for (int x=0; x<w; ++x) {
            *ptr++ = (unsigned char)x;
            *ptr++ = (unsigned char)y;
            *ptr++ = 128;
        }
    }
    printf("prepare OK\n");
    writePng(fp, data, w, h);
    fclose(fp);
}

void readImage(const string& input_path, cv::Mat& img) {
    img = cv::imread(input_path);    
    printf("img(%s) read finish\n", input_path.c_str());
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: ./image_process -i <input_path>\n");
        return -1;
    }
    string input_path(argv[1]);
    cv::Mat img, gray_img;
    // read image
    readImage(input_path, img);
    // transfrom gray-scale image
    cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);
    cv::Mat equalized_img;
    cv::imwrite("before_equailze.png", gray_img);
    cv::equalizeHist(gray_img, equalized_img);
    cv::imwrite("after_equalize.png", equalized_img);
    return 0;
}
