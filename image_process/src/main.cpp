#include "png.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <stdio.h>

using std::string;


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
    img = cv::imread(input_path, -1);    
    printf("img(%s) read finish\n", input_path.c_str());
}

void naiveHist(const cv::Mat& src, cv::Mat& dst) {
    int H = src.rows;
    int W = src.cols;
    dst = cv::Mat::zeros(H, W, CV_8UC1);
    int value_table[256] = {0};
    float proba_table[256] = {0.};
    float accu_table[256] = {0.};
    for (int h=0; h<H; ++h) {
        for (int w=0; w<W; ++w) {
            value_table[src.at<unsigned char>(h, w)]++;
        }
    }
    int max_times = 0;
    for (int i=0; i<256; ++i) {
        if (value_table[i] > max_times) {
            max_times = value_table[i];
        }
    }
    int sum = H*W;
    for (int i=0; i<256; ++i) {
        proba_table[i] = value_table[i]*1.0 / sum;
    }
    accu_table[0] = proba_table[0];
    for (int i=1; i<256; ++i) {
        accu_table[i] = accu_table[i-1] + proba_table[i];
    }
    for (int h=0; h<H; ++h) {
        for (int w=0; w<W; ++w) {
            dst.at<unsigned char>(h, w) = (unsigned char)(accu_table[src.at<unsigned char>(h, w)] * 255.);
        }
    }
}

void equalizeHist(const cv::Mat& gray_img) {
    cv::Mat equalized_img, naive_equalized_img;
    cv::equalizeHist(gray_img, equalized_img);
    naiveHist(gray_img, naive_equalized_img);
    cv::imwrite("eq.png", naive_equalized_img);

    cv::Mat concat_opencv_img, concat_naive_img;
    cv::hconcat(gray_img, equalized_img, concat_opencv_img);
    cv::hconcat(concat_opencv_img, naive_equalized_img, concat_naive_img);
    cv::imwrite("hist.png", concat_naive_img);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: ./image_process -i <input_path>\n");
        return -1;
    }
    string input_path(argv[2]);
    cv::Mat img, gray_img;
    // read image
    readImage(input_path, img);
    // transfrom gray-scale image
    cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);
    // equalize Histgram
    equalizeHist(gray_img);
    return 0;
}
