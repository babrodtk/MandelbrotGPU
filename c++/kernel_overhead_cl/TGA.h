#pragma once

#include <vector>

void toTGA(unsigned char* data, unsigned int nx, unsigned int ny, std::string filename);
void toTGA(float* data, unsigned int nx, unsigned int ny, std::string filename);