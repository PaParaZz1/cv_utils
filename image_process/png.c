#include "png.h"
#include <string.h>
#include <stdio.h>

#define DATA_STREAM FILE* fp
#define GET(c) c = fgetc(fp)
#define PUT(c) fputc(c, fp)
#define GETS(arr, n) for(int i=0; i<n; ++i) GET(arr[i]);
#define PUTS(arr, n) for(int i=0; i<n; ++i) PUT(arr[i]);


const unsigned t[] = { 0, 0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4, 0x4db26158, 0x5005713c,     // CRC32 Table
                       0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c, 0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c };
const unsigned char magic[] = "\x89PNG\r\n\32\n";

void readPng(DATA_STREAM, const unsigned char* data, unsigned w, unsigned h) {
    char buffer[32];
    GETS(buffer, 8);
    //check buffer == magic
}

void writePng(DATA_STREAM, const unsigned char* data, unsigned w, unsigned h) {
    PUTS(magic, 8);
}
