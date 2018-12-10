#include "TGA.h"

#include <fstream>


void toTGA(unsigned char* data, unsigned int nx, unsigned int ny, std::string filename) {
    std::vector<unsigned char> fileBuffer;
    fileBuffer.reserve(nx * ny * 4);

    // TGA header
    fileBuffer.push_back(0);                        // id length
    fileBuffer.push_back(0);                        // color map type
    fileBuffer.push_back(10);                       // data type code
    fileBuffer.push_back(0);                        // colormap origin LSB
    fileBuffer.push_back(0);                        // colormap origin MSB
    fileBuffer.push_back(0);                        // colormap length LSB
    fileBuffer.push_back(0);                        // colormap length MSB
    fileBuffer.push_back(0);                        // color map depth
    fileBuffer.push_back(0);                        // x origin LSB
    fileBuffer.push_back(0);                        // x origin MSB
    fileBuffer.push_back(0);                        // y origin LSB
    fileBuffer.push_back(0);                        // y origin MSB
    fileBuffer.push_back(nx & 0xff);                // width LSB
    fileBuffer.push_back((nx >> 8) & 0xff);         // width MSB
    fileBuffer.push_back(ny & 0xff);                // height LSB
    fileBuffer.push_back((ny >> 8) & 0xff);         // height MSB
    fileBuffer.push_back(24);                       // bits per pixel
    fileBuffer.push_back(0);                        // image descriptor

    for (unsigned int y = 0; y<ny; y++) {
        //encode one scanline
        unsigned char* l = &data[3 * nx*y];
        unsigned char* r = &l[3 * nx];
        while (l < r) {
            // build one packet
            fileBuffer.push_back(0);     // make room for count
            fileBuffer.push_back(l[2]);  // first pixel
            fileBuffer.push_back(l[1]);
            fileBuffer.push_back(l[0]);

            // First, try to build a RLE packet
            unsigned char* c = l + 3;
            while ((c<r)
                && (c - l < 3 * 128)
                && (l[0] == c[0])
                && (l[1] == c[1])
                && (l[2] == c[2])) {
                c += 3;
            }

            if (c - l > 3) { // Something to compress, opt for RLE-packet to store repetitions
                fileBuffer[fileBuffer.size() - 4] = static_cast<unsigned char>(((c - l) / 3 - 1) | 128);
                l = c;
            }
            else { // Nothing to compress, make non-RLE-packet

                   // search until end of scanline and packet for possible RLE packet
                for (c = l + 3; (c<r) &&
                    (c - l < 3 * 128) &&
                    (!((c[-3] == c[0]) &&
                    (c[-2] == c[1]) &&
                        (c[-1] == c[2]))); c += 3) {
                    fileBuffer.push_back(c[2]);
                    fileBuffer.push_back(c[1]);
                    fileBuffer.push_back(c[0]);
                }
                // store non-RLE-packet size
                fileBuffer[fileBuffer.size() - (c - l) - 1] = static_cast<unsigned char>((c - l) / 3 - 1);
                l = c;
            }
        }
    }

    std::ofstream dump(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    dump.write(reinterpret_cast<char*>(&fileBuffer[0]), fileBuffer.size());
    dump.close();
}


void toTGA(float* data, unsigned int nx, unsigned int ny, std::string filename) {
    std::vector<unsigned char> rgb_data(nx*ny * 3);
    for (int i = 0; i < nx*ny; ++i) {
        unsigned char value = (unsigned char) (data[i] * 255);
        rgb_data[3 * i + 0] = value;
        rgb_data[3 * i + 1] = value;
        rgb_data[3 * i + 2] = value;
    }
    toTGA(rgb_data.data(), nx, ny, filename);
}