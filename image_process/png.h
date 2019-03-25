#ifndef NAIVE_PNG_H_
#define NAIVE_PNG_H_
#include <stdio.h>

void readPng(FILE* fp, const unsigned char* data, unsigned w, unsigned h);

void writePng(FILE* fp, const unsigned char* data, unsigned w, unsigned h);

#endif // NAIVE_PNG_H_
