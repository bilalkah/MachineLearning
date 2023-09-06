#pragma once
#ifndef ORIENTATION_H
#define ORIENTATION_H

#define _USE_MATH_DEFINES

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <vector>


int j1(int, int);
int j2(int, int);
int j3(int, int);


std::vector<std::vector<double>> calculate_angles(cv::Mat, int , bool);
double gauss(double, double);
double kernel_from_function(int, double(*)(double, double));
std::vector<std::vector<double>> smooth_angles(std::vector<std::vector<double>>);
std::vector<cv::Point> get_line_ends(int, int, double, double);
cv::Mat visualize_angles(cv::Mat, cv::Mat, std::vector<std::vector<double>>, int);

#endif // !ORIENTATION_H
