# include <iostream>
# include <opencv2/opencv.hpp>

using namespace cv;

cv::Mat meanFilter(int size) {
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) / (float)(size * size);
    return kernel;
}

cv::Mat sobelFilter(char axis) {
    cv::Mat kernel;
    if (axis == 'x') {
        kernel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                                           -2, 0, 2,
                                           -1, 0, 1);
    } else if (axis == 'y') {
        kernel = (cv::Mat_<float>(3, 3) << -1, -2, -1,
                                            0,  0,  0,
                                            1,  2,  1);
    } else {
        throw std::invalid_argument("Invalid axis. Use 'x' or 'y'.");
    }
    return kernel;
}

cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
    Mat dst(src.rows,src.cols,src.type());

    Mat_<float> flipped_kernel; 
    flip(kernel, flipped_kernel, -1);

    const int dx = kernel.cols / 2;
    const int dy = kernel.rows / 2;

    for (int i = 0; i<src.rows; i++) 
    {
        for (int j = 0; j<src.cols; j++) 
        {
            float tmp = 0.0f;
            for (int k = 0; k<flipped_kernel.rows; k++) 
            {
              for (int l = 0; l<flipped_kernel.cols; l++) 
              {
                int x = j - dx + l;
                int y = i - dy + k;
                if (x >= 0 && x < src.cols && y >= 0 && y < src.rows)
                    tmp += src.at<float>(y, x) * flipped_kernel.at<float>(k, l);
              }
            }
            dst.at<float>(i, j) = saturate_cast<float>(tmp);
        }
    }
    return dst.clone();
}

cv::Mat convolution2D(cv::Mat& image, cv::Mat& kernel) {
    int image_height = image.rows;
    int image_width = image.cols;
    int kernel_height = kernel.rows;
    int kernel_width = kernel.cols;
    cv::Mat output(image_height - kernel_height + 1, image_width - kernel_width + 1, CV_32S);

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            for (int k = 0; k < kernel_height; k++) {
                for (int l = 0; l < kernel_width; l++) {
                    output.at<int>(i, j) += image.at<int>(i + k, j + l) * kernel.at<int>(k, l);
                }
            }
        }
    }

    return output;
}


int main()
{
    int size = 3;
    cv::Mat output;
    cv::Mat sobel_x = sobelFilter('x');
    cv::Mat mean_kernel = meanFilter(size);
    cv::Mat image = cv::imread("/home/yst/文档/yst/深度学习手撕代码/src/test.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    output = spatialConvolution(image, sobel_x);
    cv::imwrite("/home/yst/文档/yst/深度学习手撕代码/src/dst.jpg", output);

    return 0;
}