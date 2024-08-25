import cv2
import numpy as np
import math

# image = cv2.imread('demo-LRImage.png')
image = cv2.imread('image2.png')
height,width = image.shape[0:2]
if (len(image.shape)==3):
    channels = 3
else:
    channels =1

scaling_factor =3
new_height = height*scaling_factor
new_width = width*scaling_factor

# empty array
outputImage = np.zeros((new_height,new_width,channels),dtype=np.uint8)

# mapping
for y in range(new_height):
    for x in range(new_width):
        x_orig = x/scaling_factor
        y_orig = y/scaling_factor
        # four nearest pixels
        x_one = math.floor(x_orig)
        x_two = math.ceil(x_orig)
        x_two = min(x_two,width-1)  #to handle edge cases where the value may go beyond the boundaries
        y_one = math.floor(y_orig)
        y_two = math.ceil(y_orig)
        y_two = min(y_two,height-1)  #to handle edge cases where the value may go beyond the boundaries
        # fractional distances
        d_x = x_orig - x_one
        d_y = y_orig - y_one

        # pixel values from original image
        # 11 means y=1,x=1 and so on
        I_11 = image[y_one,x_one]
        I_12 = image[y_one,x_two]
        I_21 = image[y_two,x_one]
        I_22 = image[y_two,x_two]

        # interpolating along x axis
        I_top = I_11*(1-d_x) + I_12 * d_x
        I_bottom = I_21*(1-d_x) + I_22*d_x

        # interpolating along y axis
        I_final = I_top * (1-d_y) + I_bottom*d_y
        # print(I_final)
        outputImage[y,x] = I_final

cv2.imshow('Original Image',image)
cv2.imshow('Bilinear',outputImage)
cv2.waitKey(0)
