#include "Frequency.h"


cv::Mat ridge_freq(cv::Mat image, cv::Mat mask, std::vector<std::vector<double>> orient, int block_size, int kernel_size, int minWaveLength, int maxWaveLength) {
	int rows = image.rows;
	int cols = image.cols;

	cv::Mat freq = cv::Mat::zeros(cv::Size(cols, rows), CV_8U);

	for (int row = 1; row < rows - block_size; row += block_size) {
		for (int col = 0; col < cols - block_size; col += block_size) {
			std::vector<std::vector<uint8_t>> image_block(block_size, std::vector<uint8_t>(block_size, 0));
			for (int i = row; i < row + block_size; i++) {
				for (int j = col; j < col + block_size; j++) {
					image_block[i][j] = image.at<uint8_t>(i, j);
				}
			}
			double angle_block = orient[row / block_size][col / block_size];
			if (angle_block != 0) {
				std::vector<std::vector<uint8_t>> freq_block;
				freq_block = frequest(image_block, angle_block, kernel_size, minWaveLength, maxWaveLength);
				for (int k = row; k < row + block_size; k++) {
					for (int l = col; l < col + block_size; l++) {
						freq.at<uint8_t>(k, l) = freq_block[k - row][l - col];
					}
				}
			}
		}
	}
	return freq;
}

std::vector<std::vector<uint8_t>> frequest(std::vector<std::vector<uint8_t>> image, double orientim, int kernel_size, int minWaveLength, int maxWaveLength)
{

	double cosorient = cos(2 * orientim);
	double sinorient = sin(2 * orientim);
	double block_orient = atan2(sinorient, cosorient) / 2;

	return std::vector<std::vector<uint8_t>>();
}
