#ifndef NAIVE_PNG_H_
#define NAIVE_PNG_H_
#include <stdio.h>

extern "C" { 
    void readPng(FILE* fp, const unsigned char* data, unsigned w, unsigned h);
}

extern "C" {
    void writePng(FILE* fp, const unsigned char* data, unsigned w, unsigned h);
}

#endif // NAIVE_PNG_H_
