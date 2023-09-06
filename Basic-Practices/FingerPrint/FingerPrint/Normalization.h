#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#pragma once
#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/core/mat.hpp>

uint8_t normalize_pixel(uint8_t, double, double, double, double);
cv::Mat normalize(cv::Mat, double, double);
double meanOfImg(cv::Mat);
double stdOfImg(cv::Mat);

#endif // NORMALIZATION_H


