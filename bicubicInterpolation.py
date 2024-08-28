import cv2
import numpy as np

def cubic_interpolate(p, x):
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))

def bicubic_interpolate(p, x, y):
    arr = np.zeros(4)
    for i in range(4):
        arr[i] = cubic_interpolate(p[i], x)
    return cubic_interpolate(arr, y)

image = cv2.imread('image2.png')
height, width = image.shape[:2]
scaling_factor = 3
new_height = height * scaling_factor
new_width = width * scaling_factor

output_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

for y in range(new_height):
    for x in range(new_width):
        x_orig = x / scaling_factor
        y_orig = y / scaling_factor

        x_int = int(x_orig)
        y_int = int(y_orig)

        x_diff = x_orig - x_int
        y_diff = y_orig - y_int

        neighbors = np.zeros((4, 4, 3), dtype=np.float32)
        for m in range(-1, 3):
            for n in range(-1, 3):
                xm = min(max(x_int + m, 0), width - 1)
                yn = min(max(y_int + n, 0), height - 1)
                neighbors[m + 1, n + 1] = image[yn, xm]

        for c in range(3):
            output_image[y, x, c] = np.clip(bicubic_interpolate(neighbors[:, :, c], x_diff, y_diff), 0, 255)

cv2.imshow('Original Image', image)
cv2.imshow('Bicubic Interpolation', output_image)
cv2.waitKey(0)
