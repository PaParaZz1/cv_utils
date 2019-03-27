#include "png.hpp"
#include <string.h>
#include <stdio.h>

#define DATA_STREAM FILE* fp
#define GET(u) (u = fgetc(fp))
#define PUT(u) fputc(u, fp)
#define PUT_32(u) do {PUT((u)>>24); PUT(((u)>>16)&255); PUT(((u)>>8)&255); PUT((u)&255);} while(0)
#define PUT_8C(u) do {PUT(u); c^=(u); c = (c>>4) ^ t[c&15]; c = (c>>4) ^ t[c&15];} while(0)
#define PUT_32C(u) do {PUT_8C((u)>>24); PUT_8C(((u)>>16)&255); PUT_8C(((u)>>8)&255); PUT_8C((u)&255);} while(0)
#define PUT_16CL(u) do {PUT_8C((u)&255); PUT_8C(((u)>>8)&255);} while(0)
#define PUT_8CA(u) do {PUT_8C(u); a = (a + (u)) % 65521; b = (b + a) % 65521;} while(0)
#define GETS(arr, n) for(int i=0; i<n; ++i) GET(arr[i]);
#define PUTS(arr, n) for(int i=0; i<n; ++i) PUT(arr[i]);
#define PUTS_8C(arr, n) for(int i=0; i<n; ++i) PUT_8C(arr[i]);
#define BEGIN(s, l) do {PUT_32(l); c=~0U; PUTS_8C(s, 4)} while(0)
#define END() PUT_32(~c)


const unsigned t[] = { 0, 0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4, 0x4db26158, 0x5005713c,     // CRC32 Table
                       0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c, 0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c };
const unsigned char magic[] = "\x89PNG\r\n\32\n";
const int depth = 8;
const int color = 2;
unsigned a=1, b=0, c=0, p;  // c---CRC sum

void readPng(DATA_STREAM, const unsigned char* data, unsigned w, unsigned h) {
    p = w*3 + 1;
    char buffer[32];
    GETS(buffer, 8);
    //check buffer == magic
}

void writePng(DATA_STREAM, const unsigned char* data, unsigned w, unsigned h) {
    // 8 + 13 + 2 + h*(5+w*3+1) + 4 + 3*12
    p = w*3 + 1;
    PUTS(magic, 8);
    BEGIN("IHDR", 13);
    PUT_32C(w);
    PUT_32C(h);
    PUT_8C(depth);
    PUT_8C(color);
    PUTS_8C("\0\0\0", 3);
    END();

    printf("IHDR ok\n");
    BEGIN("IDAT", 2 + h * (5+p) + 4);
    PUTS_8C("\x78\1", 2);
    for (int y=0; y<h; ++y) {
        PUT_8C(y == h-1);  // last block is 1, others is 0
        PUT_16CL(p);   // block size
        PUT_16CL(~p);
        PUT_8CA(0);
        for (int x=0; x<p-1; ++x) {
            PUT_8CA(*data);
            data++;
        }
    }
    PUT_32C((b<<16) | a);
    END();
    printf("IDAT ok\n");

    BEGIN("IEND", 0);
    END();
}

