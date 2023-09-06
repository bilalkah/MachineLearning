#include "Normalization.h"

uint8_t normalize_pixel(uint8_t x, double v0, double v, double m, double m0)
{
	uint8_t dev_coeff = (uint8_t) sqrt((v0 * (pow((uint8_t)(x - m), 2))) / v);
	if (x > m)
	{
		return m0 + dev_coeff;
	}
	return m0 - dev_coeff;
	
}

cv::Mat normalize(cv::Mat image, double m0, double v0)
{
	double m = meanOfImg(image);
	double v = pow(stdOfImg(image), 2);
	int row = image.rows;
	int col = image.cols;
	cv::Mat normalized;
	image.copyTo(normalized);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			normalized.at<uint8_t>(i, j) = normalize_pixel(image.at<uint8_t>(i, j), v0, v, m, m0);
		}
	}
	return normalized;
}

double meanOfImg(cv::Mat image)
{
	double result = 0.0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			result += image.at<uint8_t>(i, j);
		}
	}
	result = (result / (image.cols * image.rows));
	return result;
}

double stdOfImg(cv::Mat image)
{
	double result = 0.0;
	double mean = meanOfImg(image);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			result += pow(image.at<uint8_t>(i, j) - mean, 2);
		}
	}
	return (double)sqrt(result / (image.rows * image.cols - 1));
}