#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

#include "Normalization.h"
#include "Segmentation.h"
#include "Orientation.h"


int main() {


	cv::Mat image = cv::imread("img_2.png");
	if (image.empty()) {
		return -1;
	}


	cv::Mat gray_image;
	cvtColor(image, gray_image,cv::COLOR_BGR2GRAY);


	double meanImg = meanOfImg(gray_image);
	double stdImg = stdOfImg(gray_image);
	cv::Mat normalized = normalize(gray_image, 100, 100);

	int block_size = 16;
	returnData data = create_segmented_and_variance_images(normalized, block_size, 0.2);
	cv::Mat segmented_image = data.segmented_image;
	cv::Mat norm_img = data.norm_img;
	cv::Mat mask = data.mask;

	std::vector<std::vector<double>> angles = calculate_angles(normalized, block_size, false);
	cv::Mat orientation_img = visualize_angles(segmented_image, mask, angles, block_size);
	
	cv::imshow("Normalized image", normalized);

	cv::imshow("Segmented image", segmented_image);

	cv::imshow("Norm image", norm_img);
	
	cv::imshow("Orientation image", orientation_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
	
	

	
	

	

	return 0;
}