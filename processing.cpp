#include "stdafx.h"

Processing::Processing()
{
	init_masks();
}

void Processing::gamma_correction()
{
	this->output_img = this->input_img_gray_32f.clone();
	for (int y = 0; y < this->input_img_gray_32f.rows; y++) {
		for (int x = 0; x < this->input_img_gray_32f.cols; x++) {
			float color = this->input_img_gray_32f.at<float>(y ,x);		
			this->gamma_val > this->gamma_prev ? color += 0.1f : color -= 0.1f;
			this->output_img.at<float>(y, x) = color;
		}
	}

	this->gamma_prev = this->gamma_val;
}

void Processing::convolution(cv::Mat mask, float normal)
{
	int border = (mask.cols % 3 == 0) ? 1 : mask.cols % 3;

	this->output_img = this->input_img_gray_32f.clone();
	for (int y = border; y < this->input_img_gray_32f.rows - border; y++) {
		for (int x = border; x < this->input_img_gray_32f.cols - border; x++) {
			float value = 0;
			for (int i = -border; i <= border; i++) {
				for (int j = -border; j <= border; j++) {
					if (y + i >= 0 && x + j >= 0 && y + i < this->input_img_gray_32f.rows && x + j < this->input_img_gray_32f.cols) {
						value += this->input_img_gray_32f.at<float>(y + i, x + j) * mask.at<float>(i + border, j + border);
					}
				}
			}
			this->output_img.at<float>(y, x) = value * normal;
		}
	}
}

void Processing::convolution_g(cv::Mat gx, cv::Mat gy) {
	for (int y = 0; y < this->input_img_gray_32f.rows; y++) {
		for (int x = 0; x < this->input_img_gray_32f.cols; x++) {
			float g = sqrt(pow(gx.at<float>(y, x), 2) + pow(gy.at<float>(y, x), 2));
			this->output_img.at<float>(y, x) = g;
		}
	}
}


void Processing::anisotropic_diffusion(int time, double sigma, double lambda)
{
	this->output_img = this->input_img_gray_64f.clone();

#pragma omp parallel for
	for (int y = 0; y < this->input_img_gray_64f.rows; y++) {
		for (int x = 0; x < this->input_img_gray_64f.cols; x++) {
			double px = this->input_img_gray_64f.at<double>(y, x);
			if (y - 1 >= 0 && x - 1 >= 0 && y + 1 < this->input_img_gray_64f.rows && x + 1 < this->input_img_gray_64f.cols) {
				double CN = exp(-(pow(abs(px - this->input_img_gray_64f.at<double>(y - 1, x)), 2.0f) / pow(sigma, 2.0f)));
				double CS = exp(-(pow(abs(px - this->input_img_gray_64f.at<double>(y + 1, x)), 2.0f) / pow(sigma, 2.0f)));
				double CE = exp(-(pow(abs(px - this->input_img_gray_64f.at<double>(y, x + 1)), 2.0f) / pow(sigma, 2.0f)));;
				double CW = exp(-(pow(abs(px - this->input_img_gray_64f.at<double>(y, x - 1)), 2.0f) / pow(sigma, 2.0f)));;

				px = px * (1 - (lambda * (CN + CS + CE + CW))) + (lambda *
					(
						CN * this->input_img_gray_64f.at<double>(y - 1, x) +
						CS * this->input_img_gray_64f.at<double>(y + 1, x) +
						CE * this->input_img_gray_64f.at<double>(y, x + 1) +
						CW * this->input_img_gray_64f.at<double>(y, x - 1))
					);
			}
			this->output_img.at<double>(y, x) = px;
			this->input_img_gray_64f.at<double>(y, x) = px;
		}
	}
}

void Processing::dft()
{
	cv::resize(this->input_img_gray_64f, this->input_img_gray_64f, cv::Size(125, 125));
	cv::Mat source = this->input_img_gray_64f.clone();
	cv::Mat destination = cv::Mat(cv::Size(125, 125), CV_64FC2);
	double normalization = 1.0f / sqrt(source.cols * source.rows);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int k = 0; k < source.cols; k++) {
		for (int l = 0; l < source.rows; l++) {
			double real_sum = 0.0f;
			double imaginar_sum = 0.0f;

			for (int m = 0; m < source.cols; m++) {
				for (int n = 0; n < source.rows; n++) {
					double euler = 2 * M_PI * (double)(((m * k) / (double)source.rows) + ((n * l) / (double)source.cols));

					real_sum += source.at<double>(m, n) * (normalization * cos(euler));
					imaginar_sum += source.at<double>(m, n) * (normalization * -sin(euler));
				}
			}

			destination.at<cv::Vec2d>(k, l) = cv::Vec2d(real_sum, imaginar_sum);
		}
	}
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_n = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	std::cout << "Time: " << time_n.count() * 1000 << " ms" << std::endl;

	this->dft_output = destination.clone();
	cv::resize(source, source, cv::Size(256, 256));
	cv::resize(destination, destination, cv::Size(256, 256));

	
	cv::Mat real_complex = cv::Mat::zeros(destination.size(), CV_64FC1);
	cv::Mat imaginar_complex = cv::Mat::zeros(destination.size(), CV_64FC1);

	cv::Mat complex[2] = { real_complex, imaginar_complex };
	cv::split(destination, complex);

	this->amplitude = cv::Mat(cv::Size(destination.cols, destination.rows), CV_64FC1);
	this->phase = cv::Mat(cv::Size(destination.cols, destination.rows), CV_64FC1);
	this->power = cv::Mat(cv::Size(destination.cols, destination.rows), CV_64FC1);
	cv::resize(this->input_img_gray_64f, this->input_img_gray_64f, cv::Size(destination.cols, destination.rows));

	for (int k = 0; k < destination.cols; k++) {
		for (int l = 0; l < destination.rows; l++) {
			this->amplitude.at<double>(k, l) = abs(sqrt(pow(complex[Complex::real].at<double>(k, l), 2) + pow(complex[Complex::img].at<double>(k, l), 2)));
		}
	}

	for (int k = 0; k < destination.cols; k++) {
		for (int l = 0; l < destination.rows; l++) {
			this->phase.at<double>(k, l) = atan(complex[Complex::real].at<double>(k, l) / complex[Complex::img].at<double>(k, l));
		}
	}

	for (int k = 0; k < destination.cols; k++) {
		for (int l = 0; l < destination.rows; l++) {
			this->power.at<double>(k, l) = pow(this->amplitude.at<double>(k, l), 2);
		}
	}

	cv::log(this->amplitude, this->amplitude);
	cv::log(this->power, this->power);

	cv::normalize(this->amplitude, this->amplitude, 0, 1, CV_MINMAX);

	int center_x = this->amplitude.cols / 2;
	int center_y = this->amplitude.rows / 2;
	int cx = this->power.cols / 2;
	int cy = this->power.rows / 2;

	cv::Mat q1(this->amplitude, cv::Rect(0, 0, center_x, center_y));
	cv::Mat q2(this->amplitude, cv::Rect(center_x, 0, center_x, center_y));
	cv::Mat q3(this->amplitude, cv::Rect(0, center_y, center_x, center_y));
	cv::Mat q4(this->amplitude, cv::Rect(center_x, center_y, center_x, center_y));

	cv::Mat swap;

	q1.copyTo(swap);
	q4.copyTo(q1);
	swap.copyTo(q4);

	q2.copyTo(swap);
	q3.copyTo(q2);
	swap.copyTo(q3);

	cv::Mat qA(this->power, cv::Rect(0, 0, cx, cy));
	cv::Mat qB(this->power, cv::Rect(cx, 0, cx, cy));
	cv::Mat qC(this->power, cv::Rect(0, cy, cx, cy));
	cv::Mat qD(this->power, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;

	qA.copyTo(tmp);
	qD.copyTo(qA);
	tmp.copyTo(qD);

	qB.copyTo(tmp);
	qC.copyTo(qB);
	tmp.copyTo(qC);
}


void Processing::inverse_dft()
{
	cv::resize(this->dft_output, this->dft_output, cv::Size(125, 125));
	cv::Mat destination = cv::Mat(cv::Size(125, 125), CV_64FC1);

	cv::Mat real_complex = cv::Mat::zeros(this->dft_output.size(), CV_64FC1);
	cv::Mat imaginar_complex = cv::Mat::zeros(this->dft_output.size(), CV_64FC1);

	cv::Mat complex[2] = { real_complex, imaginar_complex };
	cv::split(this->dft_output, complex);

	double normalization = 1.0f / sqrt(this->dft_output.cols * this->dft_output.rows);

#pragma omp parallel for
	for (int k = 0; k < this->dft_output.cols; k++) {
		for (int l = 0; l < this->dft_output.rows; l++) {
			double real_sum = 0.0f;
			double imaginar_sum = 0.0f;

			for (int m = 0; m < this->dft_output.cols; m++) {
				for (int n = 0; n < this->dft_output.rows; n++) {
					double euler = 2 * M_PI * (double)(((m * k) / (double)this->dft_output.rows) + ((n * l) / (double)this->dft_output.cols));

					real_sum += complex[Complex::real].at<double>(m, n) * (normalization * cos(euler));
					imaginar_sum += complex[Complex::img].at<double>(m, n) * (normalization * -sin(euler));
				}
			}

			destination.at<double>(k, l) = real_sum + imaginar_sum;
		}
	}

	this->output_img = destination.clone();
	cv::resize(this->output_img, this->output_img, cv::Size(256, 256));
	cv::resize(this->dft_output, this->dft_output, cv::Size(256, 256));
}

void Processing::filter_frequency_domain()
{
}

void Processing::lens_distortion_removal(double K1, double K2)
{
	std::cout << "Distortion Removal - K1(" << K1 << "), K2(" << K2 << ")" << std::endl;
	int s_y = this->input_img_gray_64f.rows;
	int s_x = this->input_img_gray_64f.cols;
	this->output_img = cv::Mat(cv::Size(s_x, s_y), CV_64FC1);
	K1 *= 0.01f;
	K2 *= 0.01f;
	int u = this->output_img.cols / 2;
	int v = this->output_img.rows / 2;
	double R = sqrt(pow(u, 2) + pow(v, 2));

#pragma omp parallel for
	for (int y = 0; y < this->output_img.rows; y++) {
		for (int x = 0; x < this->output_img.cols; x++) {
			double l_x = (x - u) / R;
			double l_y = (y - v) / R;
			double p = pow(l_x, 2) + pow(l_y, 2);
			double res = 1 + (K1 * p) + (K2 * pow(p, 2));
			double d_x = (x - u) * pow(res, -1) + u;
			double d_y = (y - v) * pow(res, -1) + v;

			if (K1 > 0 || K2 > 0) {
				double x_f = floor(d_x);
				double x_c = ceil(d_x);
				double y_f = floor(d_y);
				double y_c = ceil(d_y);

				double interpol_1 = ((x_c - d_x) / (x_c - x_f)) * this->input_img_gray_64f.at<double>(y_f, x_f) +
					((d_x - x_f) / (x_c - x_f)) * this->input_img_gray_64f.at<double>(y_f, x_c);
				double interpol_2 = ((x_c - d_x) / (x_c - x_f)) * this->input_img_gray_64f.at<double>(y_c, x_f) +
					((d_x - x_f) / (x_c - x_f)) * this->input_img_gray_64f.at<double>(y_c, x_c);
				this->output_img.at<double>(y, x) = ((y_c - d_y) / (y_c - y_f)) * interpol_1 + ((d_y - y_f) / (y_c - y_f)) * interpol_2;
			}
			else {
				this->output_img.at<double>(y, x) = this->input_img_gray_64f.at<double>(d_y, d_x);
			}
		}
	}
}

void Processing::histogram_equalization()
{
	cv::resize(this->input_img_gray, this->input_img_gray, cv::Size(this->input_img_gray.cols / 2, this->input_img_gray.rows / 2));
	this->output_img = this->input_img_gray.clone();
	int hist[256];
	int cumul[256];
	double ps[256];
	int final[256];
	int size = this->input_img_gray.rows * this->input_img_gray.cols;

#pragma omp parallel for
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
		ps[i] = 0;
	}

#pragma omp parallel for
	for (int y = 0; y < this->input_img_gray.rows; y++)
		for (int x = 0; x < this->input_img_gray.cols; x++)
			hist[(int)this->input_img_gray.at<uchar>(y, x)]++;

	cumul[0] = hist[0];
	for (int i = 1; i < 256; i++) {
		cumul[i] = hist[i] + cumul[i - 1];
	}
	for (int i = 1; i < 256; i++) {
		cumul[i] = cvRound((int)cumul[i] * (255.0/size));
		ps[cumul[i]] += (double)hist[i] / size;
	}

#pragma omp parallel for
	for (int i = 0; i < 256; i++)
		final[i] = cvRound(ps[i] * 255);

#pragma omp parallel for
	for (int y = 0; y < this->output_img.rows; y++)
		for (int x = 0; x < this->output_img.cols; x++)
			this->output_img.at<uchar>(y, x) = cv::saturate_cast<uchar>(cumul[this->input_img_gray.at<uchar>(y, x)]);

	this->histogram_output = calculate_histogram(final);
	std::cout << "Finished ..." << std::endl;
}

void Processing::projective_transform(cv::Mat &put)
{
	std::cout << "Starting Projective Transform ..." << std::endl;
	cv::Mat imgA = this->input_img.clone();
	cv::Mat imgB = put.clone();
	Point a1, a2, a3, a4, b1, b2, b3, b4;
	a1.x = 0, a1.y = 0;
	a2.x = 323, a2.y = 0;
	a3.x = 323, a3.y = 215;
	a4.x = 0, a4.y = 215;
	b1.x = 69, b1.y = 107;
	b2.x = 227, b2.y = 76;
	b3.x = 228, b3.y = 122;
	b4.x = 66, b4.y = 134;
	Point A[4] = {a1, a2, a3, a4};
	Point B[4] = {b1, b2, b3, b4};

	/*cv::Mat p = (cv::Mat_<float>(3, 3));
	cv::Mat coords = (cv::Mat_<float>(3, 1));
	cv::Mat h_coords = (cv::Mat_<float>(3, 1));
	*/
}

void Processing::back_projection() {

}

void Processing::laplace_operator() {
	this->output_img = this->input_img_gray_32f.clone();
	cv::GaussianBlur(output_img, output_img, cv::Size(3, 3), 0, 0);

	for (int y = 1; y < this->input_img_gray_32f.rows - 1; y++) {
		for (int x = 1; x < this->input_img_gray_32f.cols - 1; x++) {
			float fxx = input_img_gray_32f.at<float>(y, x - 1) - 2 * input_img_gray_32f.at<float>(y, x) + input_img_gray_32f.at<float>(y, x + 1);
			float fyy = input_img_gray_32f.at<float>(y - 1, x) - 2 * input_img_gray_32f.at<float>(y, x) + input_img_gray_32f.at<float>(y + 1, x);
			this->output_img.at<float>(y, x) = abs(fxx + fyy);
		}
	}
}

void Processing::laplace_operator_color() {
	this->output_img_color = cv::Mat(cv::Size(input_img_gray_32f.cols, input_img_gray_32f.rows), CV_32FC3);
	cv::GaussianBlur(output_img_color, output_img_color, cv::Size(3, 3), 0, 0);

	for (int y = 1; y < this->input_img_gray_32f.rows - 1; y++) {
		for (int x = 1; x < this->input_img_gray_32f.cols - 1; x++) {
			float fxx = input_img_gray_32f.at<float>(y, x - 1) - 2 * input_img_gray_32f.at<float>(y, x) + input_img_gray_32f.at<float>(y, x + 1);
			float fyy = input_img_gray_32f.at<float>(y - 1, x) - 2 * input_img_gray_32f.at<float>(y, x) + input_img_gray_32f.at<float>(y + 1, x);
			if ((fxx + fyy) > 0) {
				this->output_img_color.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 1, 0);
			}

			if ((fxx + fyy) < 0) {
				this->output_img_color.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 1);
			}
			this->output_img_color.at<float>(y, x) = abs(this->output_img_color.at<float>(y, x));
		}
	}
}

void Processing::canny_edge_detection(float t1, float t2) {

	convolution(masks_data[3], masks_normal[3]);
	cv::Mat gx = this->output_img;
	convolution(masks_data[4], masks_normal[4]);
	cv::Mat gy = this->output_img;

	this->output_img = cv::Mat(this->input_img_gray_32f.rows, this->input_img_gray_32f.cols, this->input_img_gray_32f.type());
	cv::GaussianBlur(output_img, output_img, cv::Size(3, 3), 0, 0);

	for (int y = 1; y < this->input_img_gray_32f.rows - 1; y++) {
		for (int x = 1; x < this->input_img_gray_32f.cols - 1; x++) {
			float e_plus, e_minus;
			float theta = atan2(gy.at<float>(y, x), gx.at<float>(y, x)) + (M_PI/2);
			float alpha = tan(theta);
			float value = input_img_gray_32f.at<float>(y, x);

			if (theta > 0 && theta <= 1 * M_PI / 4 || theta > 4 * M_PI / 4 && theta <= 5 * M_PI / 4) {
				e_plus = alpha * this->input_img_gray_32f.at<float>(y + 1, x + 1) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y, x + 1));
				e_minus = alpha * this->input_img_gray_32f.at<float>(y - 1, x - 1) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y, x - 1));
			}
			if (theta > 1 * M_PI / 4 && theta <= 2 * M_PI / 4 || theta > 5 * M_PI / 4 && theta <= 6 * M_PI / 4) {
				e_plus = alpha * this->input_img_gray_32f.at<float>(y + 1, x) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y + 1, x + 1));
				e_minus = alpha * this->input_img_gray_32f.at<float>(y - 1, x) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y - 1, x - 1));
			}
			if (theta > 2 * M_PI / 4 && theta <= 3 * M_PI / 4 || theta > 6 * M_PI / 4 && theta <= 7 * M_PI / 4) {
				e_plus = alpha * this->input_img_gray_32f.at<float>(y + 1, x - 1) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y + 1, x));
				e_minus = alpha * this->input_img_gray_32f.at<float>(y - 1, x + 1) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y - 1, x));
			}
			if (theta > 3 * M_PI / 4 && theta <= 4 * M_PI / 4 || theta > 7 * M_PI / 4 && theta <= 8 * M_PI / 4) {
				e_plus = alpha * this->input_img_gray_32f.at<float>(y, x - 1) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y + 1, x - 1));
				e_minus = alpha * this->input_img_gray_32f.at<float>(y, x + 1) + ((1 - alpha) * this->input_img_gray_32f.at<float>(y - 1, x + 1));
			}

			if (e_minus < this->input_img_gray_32f.at<float>(y, x) && this->input_img_gray_32f.at<float>(y, x) > e_plus) {
				this->output_img.at<float>(y, x) = this->input_img_gray_32f.at<float>(y, x);
			}
		}
	}

	cv::Point lookForMatrix[]{
		cv::Point(1, 1),  cv::Point(1, 0),  cv::Point(1, -1),
		cv::Point(0, 1),					cv::Point(0, -1),
		cv::Point(-1, 1), cv::Point(-1, 0), cv::Point(-1, 1)
	};
	int lookForMatrixSize = 8;
	std::vector<cv::Point> points;
	cv::Mat destination = cv::Mat(this->output_img.rows, this->output_img.cols, this->output_img.type());

	for (int y = 0; y < this->output_img.rows; y++) {
		for (int x = 0; x < this->output_img.cols; x++) {
			if (this->output_img.at<float>(y, x) > t2) {
				points.push_back(cv::Point(x, y));

				while (!points.empty()) {
					cv::Point point = points[points.size() - 1];
					points.pop_back();

					if (point.x > 0 && point.y > 0 && point.x < this->output_img.cols - 1 && point.y < this->output_img.rows - 1) {
						if (destination.at<float>(point) > 0) {	
							continue;
						}

						for (int i = 0; i < lookForMatrixSize; i++) {
							cv::Point pointToCheck = point + lookForMatrix[i];

							if (this->output_img.at<float>(pointToCheck) >= t1 && this->output_img.at<float>(pointToCheck) <= t2 && destination.at<float>(y, x) == 0) {
								points.push_back(pointToCheck);
							}
						}

						destination.at<float>(point) = 1;	
					}
				}
			}
		}
	}
	this->output_img = destination;
}

void Processing::set_gamma(int gamma_val, int gamma_prev)
{
	this->gamma_val = gamma_val;
	this->gamma_prev = gamma_prev;
}

void Processing::set_input(cv::Mat input)
{
	this->input_img = input.clone();
	if (input.channels() == 3) {
		cv::cvtColor(input.clone(), this->input_img_gray, CV_BGR2GRAY);
	}
	else {
		this->input_img_gray = input.clone();
	}
	this->input_img_gray.convertTo(this->input_img_gray_32f, CV_32FC2, 1.0 / 255.0);
	this->input_img_gray.convertTo(this->input_img_gray_64f, CV_64FC1, 1.0 / 255.0, 0);
	init_histogram();
}

cv::Mat Processing::get_input()
{
	return this->input_img;
}

cv::Mat Processing::get_input_gray()
{
	return this->input_img_gray;
}

cv::Mat Processing::get_input_gray_32f()
{
	return this->input_img_gray_32f;
}

cv::Mat Processing::get_input_gray_64f()
{
	return this->input_img_gray_64f;
}

cv::Mat Processing::get_output()
{
	return this->output_img;
}

cv::Mat Processing::get_output_color()
{
	return this->output_img_color;
}

cv::Mat Processing::get_amplitude()
{
	return this->amplitude;
}

cv::Mat Processing::get_phase()
{
	return this->phase;
}

cv::Mat Processing::get_power()
{
	return this->power;
}

cv::Mat Processing::get_dft_output()
{
	return this->dft_output;
}

void Processing::init_masks()
{
	masks_data = new cv::Mat[5];
	masks_normal = new float[5];

	cv::Mat mask = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	float n = 1.0 / 9.0;
	masks_data[0] = mask;
	masks_normal[0] = n;

	cv::Mat mask1 = (cv::Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
	n = 1.0 / 16.0;
	masks_data[1] = mask1;
	masks_normal[1] = n;

	cv::Mat mask2 = (cv::Mat_<float>(5, 5) << 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1);
	n = 1.0 / 256.0;
	masks_data[2] = mask2;
	masks_normal[2] = n;

	cv::Mat mask_gx = (cv::Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
	n = 1.0;
	masks_data[3] = mask_gx;
	masks_normal[3] = n;

	cv::Mat mask_gy = (cv::Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	n = 1.0;
	masks_data[4] = mask_gy;
	masks_normal[4] = n;
}

cv::Mat Processing::get_input_hist() {
	return this->histogram_input;
}

cv::Mat Processing::get_output_hist() {
	return this->histogram_output;
}

void Processing::init_histogram() {
	int hist[256];
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}
	for (int y = 0; y < this->input_img_gray.rows; y++)
		for (int x = 0; x < this->input_img_gray.cols; x++)
			hist[(int)this->input_img_gray.at<uchar>(y, x)]++;

	this->histogram_input = calculate_histogram(hist);
}

cv::Mat Processing::calculate_histogram(int * histogram) {
	int hist[256];
	for (int i = 0; i < 256; i++)
	{
		hist[i] = histogram[i];
	}
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((int)hist_w / 256);

	cv::Mat histImage(hist_h , hist_w, CV_8UC1, cv::Scalar(255, 255, 255));

	int max = hist[0];
	for (int i = 1; i < 256; i++) {
		if (max < hist[i]) {
			max = hist[i];
		}
	}
	for (int i = 0; i < 256; i++) {
		hist[i] = ((double)hist[i] / max)*histImage.rows;
	}
	for (int i = 0; i < 256; i++)
	{
		line(histImage, cv::Point(bin_w*(i), hist_h),
			cv::Point(bin_w*(i), hist_h - hist[i]),
			cv::Scalar(0, 0, 0), 1, 8, 0);
	}
	return histImage;
}

void Processing::write_image(cv::Mat img, const char * path) {
	try {
		cv::Mat send = img;
		send.convertTo(send, CV_8UC3, 255.0);
		cv::imwrite(path, send);
		std::cout << "(" << path << ")" << " write - OK" << std::endl;
	}
	catch (int e) {
		std::cout << "(" << path << ")" << " write - ERROR" << std::endl;
	}
}