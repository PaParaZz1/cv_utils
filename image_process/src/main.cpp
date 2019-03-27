#include "png.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <stdio.h>

using std::string;


void writePngInterface(const cv::Mat& src) {
    FILE* fp;
    fp = fopen("own.png", "w");
    int H = src.rows;
    int W = src.cols;
    unsigned char data[H*W*3];
    for (int k=0; k<3; ++k) {
        for (int h=0; h<H; ++h) {
            for (int w=0; w<W; ++w) {
                data[h*W*3+w*3+k] = src.at<cv::Vec3b>(h, w)[k];
            }
        }
    }
    writePng(fp, data, W, H);
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

    cv::Mat concat_opencv_img, concat_naive_img;
    cv::hconcat(gray_img, equalized_img, concat_opencv_img);
    cv::hconcat(concat_opencv_img, naive_equalized_img, concat_naive_img);
    cv::imwrite("hist.png", concat_naive_img);
}

template <typename T>
T clamp(T value, T min_limit, T max_limit) {
    if (value < min_limit) {
        return min_limit;
    } else if (value > max_limit) {
        return max_limit;
    } else {
        return value;
    }
}

void grayLinearTransform(const cv::Mat& gray_img) {
    int H = gray_img.rows;
    int W = gray_img.cols;
    cv::Mat result_img = cv::Mat::zeros(H, W, CV_8UC1);
    float k = 1.1;
    float b = 3.14;

    // transform
    for (int h=0; h<H; ++h) {
        for (int w=0; w<W; ++w) {
            unsigned char origin_value = gray_img.at<unsigned char>(h, w);
            unsigned char transform_value = static_cast<unsigned char>(clamp<float>(k*origin_value+b, 0., 255.));
            result_img.at<unsigned char>(h, w) = transform_value;
        }
    }
    cv::Mat concat_img;
    cv::hconcat(gray_img, result_img, concat_img);
    cv::imwrite("gray_linear_transform.png", concat_img);
}

void grayStretch(const cv::Mat& gray_img) {
    const unsigned min_limit = 40;
    const unsigned max_limit = 240;
    int H = gray_img.rows;
    int W = gray_img.cols;
    cv::Mat result_img = cv::Mat::zeros(H, W, CV_8UC1);
    // img statistics
    int value_table[256] = {0};
    int min_range, max_range;
    for (int h=0; h<H; ++h) {
        for (int w=0; w<W;++w) {
            value_table[gray_img.at<unsigned char>(h, w)]++;
        }
    }
    int sum = H*W;
    float proba_sum = 0.;
    for (int i=0; i<256; ++i) {
        proba_sum += value_table[i]*1.0/sum;
        if (proba_sum <= 0.1) {
            min_range = i;
        } else if (proba_sum <= 0.9) {
            max_range = i;
        }
    }
    if (max_range <= min_limit || min_range >= max_limit) {
        printf("%d,%d\n", min_range, max_range);
        fprintf(stderr, "terrible img pixels range\n");
        return;
    }

    // transform
    for (int h=0; h<H; ++h) {
        for (int w=0; w<W; ++w) {
            unsigned char origin_value = gray_img.at<unsigned char>(h, w);
            unsigned char transform_value;
            if (origin_value < min_range) {
                transform_value = (unsigned char)((min_limit*1.0/min_range) * origin_value); 
            } else if (origin_value > max_range) {
                transform_value = (unsigned char)(((255-max_limit)*1.0/(255-max_range)) * (origin_value - max_range) + max_limit);
            } else {
                transform_value = (unsigned char)(((max_limit-min_limit)*1.0/(max_range-min_range)) * (origin_value - min_range) + min_limit);
            }
            result_img.at<unsigned char>(h, w) = transform_value;
        }
    }    
    cv::Mat concat_img;
    cv::hconcat(gray_img, result_img, concat_img);
    cv::imwrite("gray_stretch.png", concat_img);
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
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    // equalize Histgram
    equalizeHist(gray_img);
    // gray linear transform
    grayLinearTransform(gray_img);
    // gray stretch
    grayStretch(gray_img);
    // use own png writer
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    writePngInterface(rgb_img);
    return 0;
}
