#include "Segmentation.h"

cv::Mat normalise(cv::Mat image) 
{
	uint8_t mean = (uint8_t)meanOfImg(image);
	uint8_t std = (uint8_t)stdOfImg(image);
	cv::Mat normalized;
	image.copyTo(normalized);
	for (int i = 0; i < normalized.rows; i++) {
		for (int j = 0; j < normalized.cols; j++) {
			normalized.at<uint8_t>(i, j) = (image.at<uint8_t>(i, j) - mean) / std;
		}
	}
	return normalized;
}

returnData create_segmented_and_variance_images(cv::Mat image, int w, double threshold=0.2)
{
	int row = (image).rows;
	int col = (image).cols;
	threshold = stdOfImg(image) * threshold;

	cv::Mat image_variance;
	cv::Mat segmented_image;
	(image).copyTo(segmented_image);
	image_variance = cv::Mat::zeros(row, col, CV_8U);
	cv::Mat mask = cv::Mat::ones(row, col, CV_8U);

	for (int i = 0; i < row; i += w) {
		for (int j = 0; j < col; j += w) {
			int minRow = std::min(i + w, row);
			int minCol = std::min(j + w, col);
			uint8_t std = stdBlock(image, i, j, minRow, minCol);

			for (int k = i; k < minRow; k++) {
				for (int l = j; l < minCol; l++) {
					image_variance.at<uint8_t>(k, l) = std;
				}
			}
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (image_variance.at<uint8_t>(i, j) < (uint8_t)threshold) {
				mask.at<uint8_t>(i, j) = 0;
			}
		}
	}

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(w * 2, w * 2));
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			segmented_image.at<uint8_t>(i, j) *= mask.at<uint8_t>(i, j);
		}
	}

	image = normalise(image);
	
	double mean_val = 0.0;
	int count = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (mask.at<uint8_t>(i, j) == 0) {
				mean_val += image.at<uint8_t>(i, j);
				count++;
			}
		}
	}
	mean_val /= count;

	double std_val = 0.0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (mask.at<uint8_t>(i, j) == 0) {
				std_val += pow(image.at<uint8_t>(i, j) - mean_val, 2);
			}
		}
	}
	std_val = sqrt(std_val / (count - 1));

	cv::Mat norm_img;
	image.copyTo(norm_img);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			norm_img.at<uint8_t>(i, j) = (image.at<uint8_t>(i, j) - (uint8_t)mean_val) / (std_val);
		}
	}
	returnData data;
	data.segmented_image = segmented_image;
	data.norm_img = norm_img;
	data.mask = mask;

	return data;
}



uint8_t stdBlock(cv::Mat block, int i, int j, int row, int col) {
	double mean = 0.0;
	double std = 0.0;
	for (int k = i ; k < row; k++) {
		for (int l = j ; l < col; l++) {
			mean += (block).at<uint8_t>(k, l);
		}
	}
	mean /= ((row - i) * (col - j));
	for (int k = i; k < row; k++) {
		for (int l = j; l < col; l++) {
			std += pow((block).at<uint8_t>(k, l) - mean, 2);
		}
	}
	std = sqrt(std/ ((row - i) * (col - j) - 1));
	
	return (uint8_t)std;
}
