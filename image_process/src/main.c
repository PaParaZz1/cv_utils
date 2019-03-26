#include "png.h"
#include <stdio.h>


int main() {
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
