import cv2
import numpy as np

def sobel_filter(axis):
    if axis == 'x':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    elif axis == 'y':
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

def convolution2D(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

if __name__ == "__main__":
    
    sobel_x = sobel_filter('x')  # 生成 Sobel X 方向滤波器
    sobel_y = sobel_filter('y')  # 生成 Sobel Y 方向滤波器

    kernel = np.ones((3, 3), dtype=np.float32) / (3 * 3)
    print(kernel.shape)

    image = cv2.imread("./test.jpg", )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)

    output = convolution2D(image, sobel_x)
    print(output.shape)

    cv2.imwrite("dst.jpg", output)
