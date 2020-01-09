// ano1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int main(int argc, char* argv[])
{

	cv::Mat img_valve = cv::imread("images/valve.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_lena = cv::imread("images/lena.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_flag = cv::imread("images/flag.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_vsb = cv::imread("images/vsb.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_window = cv::imread("images/distorted_window.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_panorama = cv::imread("images/distorted_panorama.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_eq = cv::imread("images/uneq.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_eq2 = cv::imread("images/uneq_map.jpg", CV_LOAD_IMAGE_COLOR);

	Processing p;


	/****
	*	ANO
	****/

	// Convolution - sobel
	/*cv::Mat gx, gy;
	p.set_input(img_valve);
	p.convolution(p.masks_data[3], p.masks_normal[3]);
	gx = p.get_output();
	p.convolution(p.masks_data[4], p.masks_normal[4]);
	gy = p.get_output();
	cv::imshow("Original", p.get_input_gray_32f());
	cv::imshow("Convolution Gx", gx);
	cv::imshow("Convolution Gy", gy);
	p.convolution_g(gx, gy);
	cv::imshow("Edges", p.get_output());
	cv::moveWindow("Original", 10, 10);
	cv::moveWindow("Convolution Gx", 10 + gx.cols, 10);
	cv::moveWindow("Convolution Gy", 10, 10 + gx.rows);
	cv::moveWindow("Edges", 10 + gx.cols, 10 + gx.rows);*/

	// Laplace operator
	/*p.set_input(img_valve);
	p.laplace_operator();
	p.laplace_operator_color();
	cv::imshow("Original", p.get_input());
	cv::imshow("Laplace operator", p.get_output());
	cv::imshow("Laplace operator color", p.get_output_color());
	cv::moveWindow("Original", 10, 10);
	cv::moveWindow("Laplace operator", 10 + p.get_output().cols, 10);
	cv::moveWindow("Laplace operator color", 10, 10 + p.get_output().rows);*/

	// Canny Edge - Double Thresholding
	/*int slider1 = 10;
	int slider2 = 10;
	cv::namedWindow("Canny");
	cv::namedWindow("Original");
	cv::createTrackbar("T1", "Canny", &slider1, 100);
	cv::createTrackbar("T2", "Canny", &slider2, 100);
	p.set_input(img_valve);

	while (true) {	
		p.canny_edge_detection((float)slider1/100, (float)slider2/100);
		cv::imshow("Original", p.get_input());
		cv::imshow("Canny", p.get_output());
		cv::waitKey(1);
	}*/



	/****
	*	DZO
	****/

	// Gamma Correction
	/*p.set_input(img_lena);
	p.gamma_correction();
	cv::imshow("Original", p.get_input_gray_32f());
	cv::imshow("Gamma Correction", p.get_output());
	p.write_image(p.get_input_gray_32f(), "output/1_gamma_correction/input.jpg");
	p.write_image(p.get_output(), "output/1_gamma_correction/correction.jpg");*/

	// Convolution
	/*p.set_input(img_lena);
	p.convolution(p.masks_data[2], p.masks_normal[2]);
	cv::imshow("Original", p.get_input_gray_32f());
	cv::imshow("Convolution", p.get_output());
	p.write_image(p.get_input_gray_32f(), "output/2_convolution/input.jpg");
	p.write_image(p.get_output(), "output/2_convolution/convolution.jpg");*/

	// Anisotropic Diffusion
	/*p.set_input(img_lena);
	cv::namedWindow("Anisotropic Diffusion");
	cv::imshow("Original", p.get_input_gray_64f());
	for (int i = 0; i < 100; i++) {
		p.anisotropic_diffusion(i);
		cv::imshow("Anisotropic Diffusion", p.get_output());
		cv::waitKey(1);
	}
	p.set_input(img_lena);
	p.write_image(p.get_input_gray_64f(), "output/3_anisotropic_diffusion/input.jpg");
	p.write_image(p.get_output(), "output/3_anisotropic_diffusion/an_diff_output.jpg");*/

	// Discrete Fourier Transform
	/*p.set_input(img_lena);
	cv::Mat origin = p.get_input_gray_64f().clone();
	cv::resize(origin, origin, cv::Size(256, 256));
	p.dft();
	cv::imshow("Original", origin);
	cv::imshow("Amplitude", p.get_amplitude());
	cv::imshow("Phase", p.get_phase());
	cv::imshow("Power", p.get_power());
	cv::moveWindow("Original", 10, 10);
	cv::moveWindow("Amplitude", 10 + origin.cols, 10);
	cv::moveWindow("Phase", 10, 10 + origin.rows);
	cv::moveWindow("Power", 10 + origin.cols, 10 + origin.rows);
	p.write_image(p.get_input_gray_64f(), "output/4_discrete_fourier_transform/input.jpg");
	p.write_image(p.get_amplitude(), "output/4_discrete_fourier_transform/amplitude.jpg");
	p.write_image(p.get_phase(), "output/4_discrete_fourier_transform/phase.jpg");
	p.write_image(p.get_power(), "output/4_discrete_fourier_transform/power.jpg");*/

	// Inverse Discrete Fourier Transform
	/*p.set_input(img_lena);
	p.dft();
	p.inverse_dft();
	cv::imshow("Amplitude", p.get_amplitude());
	cv::imshow("Inverse DFT", p.get_output());
	p.set_input(img_lena);
	p.write_image(p.get_input_gray_64f(), "output/5_inverse_discrete_fourier_transform/input.jpg");
	p.write_image(p.get_amplitude(), "output/5_inverse_discrete_fourier_transform/amplitude.jpg");
	p.write_image(p.get_output(), "output/5_inverse_discrete_fourier_transform/inverse_dft.jpg");*/

	// Projective Transform
	/*p.set_input(img_vsb);
	p.projective_transform(img_flag);
	cv::imshow("IMG1", img_vsb);
	cv::imshow("IMG2", img_flag);
	cv::imshow("OUTPUT", p.get_output());*/

	// Lens Distortion Removal  (W,S key - K1; D,A key - K2; ENTER = finish)
	/*p.set_input(img_window);
	cv::imshow("Original", p.get_input());
	cv::moveWindow("Original", 10, 10);
	bool distorted = true;
	float k1 = 0.0f;
	float k2 = 0.0f;
	p.lens_distortion_removal(k1, k2);
	cv::imshow("Distortion Removal", p.get_output());
	cv::moveWindow("Distortion Removal", 10 + p.get_input().cols, 10);
	while (distorted){
		char key = (char) cv::waitKey();
		if (key == 119 && k1 != 100.0f){
			k1 += 1.0f;
			p.lens_distortion_removal(k1, k2);
			cv::imshow("Distortion Removal", p.get_output());
			continue;
		}
		if (key == 100 && k2 != 100.0f){
			k2 += 1.0f;
			p.lens_distortion_removal(k1, k2);
			cv::imshow("Distortion Removal", p.get_output());
			continue;
		}
		if (key == 115 && k1 != 0.0f) {
			k1 -= 1.0f;
			p.lens_distortion_removal(k1, k2);
			cv::imshow("Distortion Removal", p.get_output());
			continue;
		}
		if (key == 97 && k2 != 0.0f) {
			k2 -= 1.0f;
			p.lens_distortion_removal(k1, k2);
			cv::imshow("Distortion Removal", p.get_output());
			continue;
		}
		if (key == 13){
			std::cout << "Finished" << std::endl;
			distorted = false;
			p.write_image(p.get_input_gray_64f(), "output/7_lens_distortion/input.jpg");
			p.write_image(p.get_output(), "output/7_lens_distortion/distortion_removed.jpg");
			break;
		}
	}*/

	// Histogram Equalization
	/*p.set_input(img_eq2);
	p.histogram_equalization();
	cv::imshow("Original", p.get_input_gray());
	cv::imshow("Equalization", p.get_output());
	cv::imshow("Original histogram", p.get_input_hist());
	cv::imshow("Equalized histogram", p.get_output_hist());
	cv::moveWindow("Original", 10, 10);
	cv::moveWindow("Equalization", 10 + p.get_input_gray().cols, 10);
	cv::moveWindow("Original histogram", 10, 40 + p.get_input_gray().rows);
	cv::moveWindow("Equalized histogram", 10 + p.get_input_gray().cols, 40 + p.get_input_gray().rows);
	p.write_image(p.get_input_gray_64f(), "output/8_histogram_equalization/input.jpg");
	try {
		cv::imwrite("output/8_histogram_equalization/equalized_img.jpg", p.get_output());
		std::cout << "(output/8_histogram_equalization/equalized_img.jpg) write - OK" << std::endl;
	}
	catch (int e) {
		std::cout << "(output/8_histogram_equalization/equalized_img.jpg) write - ERROR" << std::endl;
	}
	p.write_image(p.get_input_hist(), "output/8_histogram_equalization/input_histogram.jpg");
	p.write_image(p.get_output_hist(), "output/8_histogram_equalization/equalized_img_histogram.jpg");*/

	cv::waitKey(0);
	return 0;
}


