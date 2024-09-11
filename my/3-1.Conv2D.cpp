# include <iostream>
# include <opencv2/opencv.hpp>

using namespace std;

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

// 使用模板类
// cv::Mat 时，每次访问元素都需要显式转换类型，这可能导致代码冗长且易出错。
//      mat1.at<float>(0, 0) = 10.0f;
// cv::Mat_<float> 由于模板参数已经确定数据类型，访问元素时不需要显式转换，代码更简洁。
//      mat1(0, 0) = 10.0f;
cv::Mat_<float> Conv2D(const cv::Mat_<float>& src, \
                        const cv::Mat_<float>& kernel, int stride) {

    cv::Mat dst(src.rows / stride + 1, src.cols / stride + 1, src.type());
    cv::Mat_<float> flipped_kernel;
    flip(kernel, flipped_kernel, -1); // filp: -1 水平与垂直翻转

    // kernel 的中心
    const int dx = kernel.cols / 2;
    const int dy = kernel.rows / 2;

    // dst 的坐标
    int w = 0;
    int h = 0;

    for (int i = 0; i < src.rows; i += stride) {
        h = 0;
        for (int j = 0; j < src.cols; j += stride) {

            // 与 py 不同, Cpp 要自己实现 矩阵乘法(TODO 熊猫老师)
            float tmp = 0;
            for (int k = 0; k < flipped_kernel.rows; ++k) {
                for (int l = 0; l < flipped_kernel.cols; ++l) {

                    // 边界检查 -> 确保 kernel 覆盖了 src(因为没有 padding)
                    int x = j - dx + l;
                    int y = i - dy + k;
                    if (x >= 0 && x < src.cols && y >= 0 && y < src.rows)
                        tmp += src.at<float>(y, x) * flipped_kernel.at<float>(k, l);
                }
            }
            cout << w << ", " << h << endl;
            dst.at<float>(w, h) = cv::saturate_cast<float>(tmp); // cv::saturate_cast: 防止溢出
            h++;
        }
        w++;
    }

    return dst.clone();
}

int main()
{
    int size = 3;
    cv::Mat output;
    cv::Mat sobel_x = sobelFilter('x');
    cv::Mat mean_kernel = meanFilter(size);
    cv::Mat image = cv::imread("/home/yst/文档/yst/深度学习手撕代码/my/test.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    output = Conv2D(image, sobel_x, 3);
    cv::imwrite("/home/yst/文档/yst/深度学习手撕代码/my/dst.jpg", output);

    return 0;
}