import cv2
import os

# downscaling_factor = 2
bsd_100_path = r'E:\semester2\research\basics\BSD100'
hr_paths = {
    '2x': os.path.join(bsd_100_path, 'image_SRF_2'),
    '3x': os.path.join(bsd_100_path, 'image_SRF_3'),
    '4x': os.path.join(bsd_100_path, 'image_SRF_4')
}
train_hr_path = r'E:\semester2\research\basics\train\HR'
train_lr_path = r'E:\semester2\research\basics\train\LR'

# downscaling images

def downscale_images(hr_folder,lr_folder,scale_factor):
    # if lr_folder does not exist
    if not os.path.exists(lr_folder):
        os.makedirs('lr_folder')

    # iterate through each image in the high-resolution folder
    for image in os.listdir(hr_folder):
        hr_image_path = os.path.join(hr_folder,image)
        lr_image_path = os.path.join(lr_folder,image)

        # reading HR image
        hr_image = cv2.imread(hr_image_path)
        # skipping an image if it cannot be read
        if hr_image is None:
            continue
        # calculating new dimensions
        new_width = hr_image.shape[1] // scale_factor
        new_height = hr_image.shape[0] // scale_factor

        # Downscaling the image
        lr_image = cv2.resize(hr_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # Saving the low-resolution image
        cv2.imwrite(lr_image_path, lr_image)


# Processing images for each scaling factor
for scale, hr_folder in hr_paths.items():
    scale_factor = int(scale[0])
    lr_folder = os.path.join(train_lr_path, f'image_SRF_{scale_factor}')
    downscale_images(hr_folder, lr_folder, scale_factor)
