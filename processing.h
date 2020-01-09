#pragma once
class Processing {
	public:
		Processing();
		void gamma_correction();
		void convolution(cv::Mat mask, float normal);
		void convolution_g(cv::Mat gx, cv::Mat gy);
		void anisotropic_diffusion(int time, double sigma = 0.015f, double lambda = 0.1f);
		void dft();
		void inverse_dft();
		void filter_frequency_domain();
		void lens_distortion_removal(double K1, double K2);
		void histogram_equalization();
		void projective_transform(cv::Mat &put);
		void back_projection();
		void laplace_operator();
		void laplace_operator_color();
		void canny_edge_detection(float t1, float t2);

		void set_gamma(int gamma_val, int gamma_prev);
		void set_input(cv::Mat input);
		void write_image(cv::Mat img, const char * path);
		cv::Mat get_input();
		cv::Mat get_input_gray();
		cv::Mat get_input_gray_32f();
		cv::Mat get_input_gray_64f();
		cv::Mat get_output();
		cv::Mat get_output_color();
		cv::Mat get_amplitude();
		cv::Mat get_phase();
		cv::Mat get_power();
		cv::Mat get_dft_output();
		cv::Mat get_input_hist();
		cv::Mat get_output_hist();
		cv::Mat * masks_data;
		float * masks_normal;

	private:
		enum Complex {
			real, img
		};
		struct Point {
			float x;
			float y;
		};
		int gamma_val = 20;
		int gamma_prev = 5;
		cv::Mat input_img;
		cv::Mat input_img_gray;
		cv::Mat input_img_gray_32f;
		cv::Mat input_img_gray_64f;
		cv::Mat output_img;
		cv::Mat output_img_color;
		cv::Mat dft_output;
		cv::Mat amplitude;
		cv::Mat phase;
		cv::Mat power;
		cv::Mat histogram_output;
		cv::Mat histogram_input;
		cv::Mat calculate_histogram(int * histogram);
		void init_masks();
		void init_histogram();
};
