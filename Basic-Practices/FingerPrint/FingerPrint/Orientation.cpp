#include "Orientation.h"


int j1(int x, int y) {
	return 2 * x * y;
}
int j2(int x, int y) {
	return pow(x, 2) - pow(y, 2);
};
int j3(int x, int y) {
	return pow(x, 2) + pow(y, 2);
};


std::vector<std::vector<double>> calculate_angles(cv::Mat image, int width, bool smooth=false) {
	int row = image.rows;
	int col = image.cols;

	cv::Mat ySobel = (cv::Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	cv::Mat xSobel = (cv::Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	std::vector<std::vector<double>> result((row - 1) / width + 1);

	cv::Mat Gx_, Gy_,im;
	image.copyTo(im);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			im.at<uint8_t>(i, j) /= 125;
		}
	}

	cv::filter2D(im, Gx_, -1, ySobel);
	cv::filter2D(im, Gy_, -1, xSobel);

	
	for (int i = 1; i < row; i += width) {
		for (int j = 1; j < col; j += width) {
			double nominator = 0;
			double denominator = 0;
			for (int k = i; k < std::min(i + width, row - 1); k++) {
				for (int l = j; l < std::min(j + width, col - 1); l++) {
					int Gx = std::round(Gx_.at<uint8_t>(k, l));
					int Gy = std::round(Gy_.at<uint8_t>(k, l));
					nominator += j1(Gx, Gy);
					denominator += j2(Gx, Gy);
				}
			}
			try {
				if (nominator || denominator) {
					double angle = (M_PI + atan2(nominator, denominator)) / 2;
					double orientation = M_PI_2 + atan2(nominator, denominator) / 2;
					result.at(int((i - 1) / width)).push_back(angle);
				}
				else {
					result.at(int((i - 1) / width)).push_back(0);
				}
			}
			catch (const std::out_of_range& oor) {
				exit(-1);
			}
			
		}
	}
	/*
	if (smooth) {
		result = smooth_angles(result);
	}
	*/
	
	return result;
}

double gauss(double x, double y) {
	double ssigma = 1.0;
	return (1 / (2 * M_PI * ssigma)) * exp(-(x * x + y * y) / (2 * ssigma));
}

std::vector<cv::Point> get_line_ends(int i, int j, int width, double tang) {
	cv::Point begin, end;
	std::vector<cv::Point> returnVector;
	if (-1 <= tang && tang <= 1) {
		begin = cv::Point(i, int((-width / 2) * tang + j + width / 2));
		end = cv::Point(i + width, int((width / 2) * tang + j + width / 2));
	}
	else {
		begin = cv::Point(int(i + width / 2 + width / (2 * tang)), j + width / 2);
		end = cv::Point(int(i + width / 2 - width / (2 * tang)), j - width / 2);
	}
	returnVector.push_back(begin);
	returnVector.push_back(end);
	return returnVector;
}

cv::Mat visualize_angles(cv::Mat image, cv::Mat mask, std::vector<std::vector<double>> angles,int width){
	int row = image.rows;
	int col = image.cols;
	cv::Mat result;

	cv::cvtColor(cv::Mat::zeros(cv::Size(col, row), CV_8UC1), result, cv::COLOR_GRAY2RGB);

	int mask_threshold = pow((width - 1), 2);
	for (int i = 1; i < col; i += width) {
		for (int j = 1; j < row; j += width) {
			double radian = 0.0;
			for (int k = (j - 1); k < j + width; k++) {
				for (int l = (i - 1); l < i + width; l++) {
					radian += mask.at<uint8_t>(k, l);
				}
			}
			if (radian > mask_threshold) {
				double tang = tan(angles[(j - 1) / width][(i - 1) / width]);
				std::vector<cv::Point> ans = get_line_ends(i, j, width, tang);
				cv::line(result, ans[0], ans[1], 150);
			}
		}
	}
	cv::resize(result, result, cv::Size(col, row));
	return result;
}