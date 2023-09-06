#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include "Normalization.h"

typedef struct returnData {
	cv::Mat segmented_image;
	cv::Mat norm_img;
	cv::Mat mask;
}returnData;

cv::Mat normalise(cv::Mat);
returnData create_segmented_and_variance_images(cv::Mat, int, double);
uint8_t stdBlock(cv::Mat , int , int , int , int );


#endif //SEGMENTATION_H