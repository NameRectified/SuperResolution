import cv2
import numpy as np
import math

image = cv2.imread('demo-LRImage.png')
# cv2.imshow('image',image)
# cv2.waitKey(0)

height,width = image.shape[0:2]
if (len(image.shape)==3):
    channels = 3
else:
    channels = 1

scaling_factor = 2
new_height = height*scaling_factor
new_width = width*scaling_factor

output_image = np.zeros((new_height,new_width,channels),dtype=np.uint8)

for y in range(new_height):
    for x in range(new_width):
        # print(x,y)
        x_orig = x/scaling_factor
        y_orig = y/scaling_factor

        x_one = x_orig-1
        x_one = max(x_one,0)
        x_two = x_orig
        x_three = x_orig+1
        x_four = x_orig+2
        x_four = min(x_four,width-1)
        y_one = y_orig-1
        y_one = max(y_one,0)
        y_two = y_orig
        y_three = y_orig+1
        y_four = y_orig+2
        y_four = min(y_four,height-1 )

        neighbors = [[image[y_one, x_one], image[y_one, x_two], image[y_one, x_three], image[y_one, x_four]],
    [image[y_two, x_one], image[y_two, x_two], image[y_two, x_three], image[y_two, x_four]],
    [image[y_three, x_one], image[y_three, x_two], image[y_three, x_three], image[y_three, x_four]],
    [image[y_four, x_one], image[y_four, x_two], image[y_four, x_three], image[y_four, x_four]]
]
        d_x = x_orig - x_two
        d_y = y_orig -y_two