import cv2
import numpy as np

def sobel_filter(axis):
    if axis == 'x':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    elif axis == 'y':
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
def Conv2D(image, kernel, stride=1):
    image_H, image_W = image.shape
    kernel_H, kernel_W = kernel.shape
    
    # y = ((H + 2*padding - K_h) / stride) + 1
    output = np.zeros((int((image_H - kernel_H) / stride + 1), 
                       int((image_W - kernel_W) / stride + 1)))

    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            start_i = i * stride
            start_j = j * stride

            output[i, j] = np.sum(image[start_i:start_i + kernel_H, 
                                        start_j:start_j + kernel_W] * kernel)
    
    return output

if __name__ == "__main__":
    
    sobel_x = sobel_filter('x')  # 生成 Sobel X 方向滤波器
    sobel_y = sobel_filter('y')  # 生成 Sobel Y 方向滤波器

    kernel = np.ones((3, 3), dtype=np.float32) / (3 * 3)
    print(kernel.shape)

    image = cv2.imread("src/test.jpg", )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)

    output = Conv2D(image, sobel_x, stride=2)
    print(output.shape)

    cv2.imwrite("dst.jpg", output)