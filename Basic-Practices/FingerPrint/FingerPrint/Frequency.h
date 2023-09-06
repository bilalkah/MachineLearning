
#ifndef FREQUENCY_H
#define FREQUENCY_H

#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include<vector>
#include <cmath>
std::vector<std::vector<uint8_t>> frequest(std::vector<std::vector<uint8_t>>, double, int, int, int);
cv::Mat ridge_freq(cv::Mat, cv::Mat, cv::Mat, int, int, int, int);

#endif // !FREQUENCY_H



