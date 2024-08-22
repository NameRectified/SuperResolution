import cv2 #for reading image
import numpy as np
import math
import matplotlib.pyplot as plt # for comparing input and output side by side
# image = cv2.imread('demo-LRImage.png')
image = cv2.imread('image2.png')
# to view the imported image
# cv2.imshow('image',lrImage)
# to show the output until you press a key (if you dont put this the output is shown very fast and close)
# cv2.waitKey(0)

# .shape consists of the number of rows(height), columns(width), number of channels present in the image
# print(lrImage.shape)
# if len==2 it means it is grayscale, if it is 3 it means it is RGB
# print(len(lrImage.shape))

# height = lrImage.shape[0]
# width = lrImage.shape[1]
height,width = image.shape[:2]
if(len(image.shape)==3):
    channels = image.shape[2]
else:
    channels = 1

# scaling factor
scaling_factor =3
# new dimensions
new_height = height*scaling_factor
new_width = width*scaling_factor

# empty output image array
outputImg = np.zeros((new_height,new_width,channels),dtype=np.uint8)
# print(outputImg)

for y in range(new_height):
    for x in range(new_width):
        # print("Y:",y,"X",x)
        mapped_x = math.floor(x/scaling_factor)
        mapped_y = math.floor(y/scaling_factor)
        # print(mapped_x,mapped_y)
        orgPixelValue = image[mapped_y,mapped_x]
        outputImg[y,x] = orgPixelValue

cv2.imshow('Original Image',image)
cv2.imshow('Nearest Neighbor',outputImg)
# cv2.waitKey(0)

# matplotlib expects rgb scheme, where as in cv2 it is bgr scheme
input_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output_image_rgb = cv2.cvtColor(outputImg, cv2.COLOR_BGR2RGB)

# Plot images
plt.figure(figsize=(10, 5))  # Adjust figure size as needed

# Input Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image_rgb)
plt.axis('on')

# Resized Image
plt.subplot(1, 2, 2)
plt.title('Resized Image')
plt.imshow(output_image_rgb)
plt.axis('on')

plt.show()