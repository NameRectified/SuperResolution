import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # Sequential class is used to build a linear stack of layers
#Conv2D is  a 2d convolution layer that performs convolution operation on 2d input data such as images.
from tensorflow.keras.layers import Conv2D
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim # for importing evaluation metrics

# Defining paths to training data
bsd_100_path = r'E:\semester2\research\basics\BSD100'
# a dictionary containing the scale of the images present in the folder and the folder locations
hr_paths = {
    '2x': os.path.join(bsd_100_path, 'image_SRF_2'),
    '3x': os.path.join(bsd_100_path, 'image_SRF_3'),
    '4x': os.path.join(bsd_100_path, 'image_SRF_4')
}
train_hr_path = r'E:\semester2\research\basics\train\HR'
train_lr_path = r'E:\semester2\research\basics\train\LR'

# function that  downscales images from the HR folder and saves them to LR folder
def downscale_images(hr_folder, lr_folder, scale_factor):
    # if the lr folder does not exist on the systeam it is created
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    # for each image in the hr_folder
    for image in os.listdir(hr_folder):
        # complete path to each iterated image
        hr_image_path = os.path.join(hr_folder, image)
        lr_image_path = os.path.join(lr_folder, image)
        # reading image using cv2
        hr_image = cv2.imread(hr_image_path)
        if hr_image is None:
            continue  # Skips if the current image cannot be read

        new_width = hr_image.shape[1] // scale_factor # dividing the width of the hr image with the scale factor
        new_height = hr_image.shape[0] // scale_factor # dividingg the height of the hr image with the scale factor

        # Resizing the high-resolution image to create a low-resolution version
        # we are using bicubic interpolation
        lr_image = cv2.resize(hr_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # saving the new downscaled image to lr folder
        cv2.imwrite(lr_image_path, lr_image)

# Processing images for each scaling factor
for scale, hr_folder in hr_paths.items():
    # we are trying to get only the number like from '2x' we just want 2 and then converting it into an integer
    scale_factor = int(scale[0])
    # creating a low resolution image folder for each of the scale factors
    lr_folder = os.path.join(train_lr_path, f'image_SRF_{scale_factor}')
    # calling the downscale_images function for each of the scale factors
    downscale_images(hr_folder, lr_folder, scale_factor)

# Function to build the SRCNN model
def build_srcnn_model(input_shape):
    model = Sequential() # Empty sequential model
    # we add layers to the sequential model using .add
    # first conv layer has 64 filters and a kernel size of 9x9 with ReLU activation and same padding
    # same padding ensures that the output has the same width and height as the input
    model.add(Conv2D(filters=64, kernel_size=9, padding='same', activation='relu', input_shape=input_shape))
    # a smaller kernel size for layer 2 (1x1) with ReLU activation and 32 filters
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='relu'))

    model.add(Conv2D(filters=1, kernel_size=5, padding='same'))
    return model

# calling the function to build the model
# the third argument shows that it can accept only images with one channel (grayscale)and remaining arguments show that any width and height is acceptable
model = build_srcnn_model((None, None, 1))  # Assuming grayscale images
# compiles the model with adam optimizer and MSE as the loss function
model.compile(optimizer='adam', loss='mean_squared_error')


# we are doing this because we want to find the smalled height and width among all images. This will be useful in ensuring that all images in the training have consistent dimensions which are required for training the model.

# Function to find the smallest dimensions of all images
def find_min_dimensions(hr_folder, lr_folder):
    min_width, min_height = float('inf'), float('inf') # minimum width and height are initialized to infinity
    # iterating through all hr images in hr folder
    for image_name in os.listdir(hr_folder):
        hr_image_path = os.path.join(hr_folder, image_name)
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_GRAYSCALE) # reading image in grayscale
        if hr_image is None:
            continue # skipping images that cannot be read

        min_width = min(min_width, hr_image.shape[1]) # updating minimum width
        min_height = min(min_height, hr_image.shape[0]) # updating maximum width

    return min_width, min_height

# Function to load data
def load_data(hr_folder, lr_folder):
    hr_images = []
    lr_images = []

    # Determining the minimum dimensions to resize all images to
    min_width, min_height = find_min_dimensions(hr_folder, lr_folder)

    for image_name in os.listdir(hr_folder):
        hr_image_path = os.path.join(hr_folder, image_name)
        lr_image_path = os.path.join(lr_folder, image_name)

        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_GRAYSCALE)
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_GRAYSCALE)

        if hr_image is None or lr_image is None:
            continue  # Skip if images are not found

        # Resize images to the minimum dimensions
        hr_image_resized = cv2.resize(hr_image, (min_width, min_height), interpolation=cv2.INTER_CUBIC)
        lr_image_resized = cv2.resize(lr_image, (min_width, min_height), interpolation=cv2.INTER_CUBIC)

        hr_images.append(hr_image_resized)
        lr_images.append(lr_image_resized)

    hr_images = np.array(hr_images).astype('float32') / 255.0
    lr_images = np.array(lr_images).astype('float32') / 255.0

    return lr_images, hr_images

# Iterate over each scaling factor to load data and train model
for scale, hr_folder in hr_paths.items():
    scale_factor = int(scale[0])
    lr_folder = os.path.join(train_lr_path, f'image_SRF_{scale_factor}')

    # Load images for each scale
    lr_images, hr_images = load_data(hr_folder, lr_folder)

    # Ensure data is loaded correctly
    print(f'Loaded {len(lr_images)} low-resolution images for scale {scale_factor}x')
    print(f'Loaded {len(hr_images)} high-resolution images for scale {scale_factor}x')

    # Train model for each scale factor if images are loaded
    if len(lr_images) > 0 and len(hr_images) > 0:
        model.fit(lr_images, hr_images, epochs=10, batch_size=16, validation_split=0.1)
    else:
        print(f"Skipping training for scale {scale_factor}x due to lack of data.")

# Save the trained model
model.save('srcnn_model.h5')

# Function to generate a super-resolved image
def generate_sr_image(lr_image_path, model):
    lr_image = cv2.imread(lr_image_path, cv2.IMREAD_GRAYSCALE)
    lr_image = lr_image.astype('float32') / 255.0
    lr_image = np.expand_dims(lr_image, axis=(0, -1))  # Add batch and channel dimensions

    sr_image = model.predict(lr_image)
    sr_image = sr_image.squeeze()  # Remove batch and channel dimensions
    sr_image = np.clip(sr_image * 255.0, 0, 255).astype('uint8')  # Scale back to 0-255

    return sr_image

# Test the model on a sample low-resolution image
test_lr_image_path = r'E:\semester2\research\basics\image2.png'
sr_image = generate_sr_image(test_lr_image_path, model)
cv2.imwrite('sr_test_image.jpg', sr_image)

# Calculate PSNR and SSIM for the generated image
# Load the corresponding high-resolution image
hr_image = cv2.imread(test_lr_image_path, cv2.IMREAD_GRAYSCALE)
if hr_image is not None:
    psnr_value = psnr(hr_image, sr_image)
    ssim_value = ssim(hr_image, sr_image, data_range=hr_image.max() - hr_image.min())

    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")
else:
    print(f"Could not read the high-resolution image at {test_lr_image_path}.")
