import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
''' General representation is FSRCNN(d,s,m)
    # The default network according to the paper is FSRCNN(56,12,4)
    # Activation function: PReLU
    # Optimizer: Stochastic Gradient Descent (SGD)
    # Channels: 1
 '''

# def fsrcnn(d=56,s=12,m=4,channels=1):
def fsrcnn(d=56,s=16,m=4,channels=3):
    # notation Conv(f,n,c)
    # 1st parameter is kernel size
    # 2nd parameter is no. of output channels
    # 3rd parameter is number of input channels. In keras the deault is 1
    # Note: The notation given in the paper is different from tensorflow, in tensorflow it is (n,(fxf),c) but c is generally taken from previous layer
    # the number of input channels for the 2nd convolutional layer, is automatically got from the previous layer, as the first layer outputs d feature maps, the second layer expects d input channels.
    inputs = tf.keras.Input(shape = (None,None,channels))

    ''' Layer 1: The layer that extracts features.
    '''
    x = Conv2D(d,(5,5),padding='same')(inputs)
    #  Applying PReLU activation function
    x = PReLU(shared_axes = [1,2])(x)

    ''' Layer 2: Shrinking layer
    '''
    x = Conv2D(s,(1,1),padding='same')(x)
    x = PReLU(shared_axes = [1,2])(x)

    '''
        Layer 3: Non linear mapping
    '''
    for i in range(m):
        x = Conv2D(s,(3,3),padding='same')(x)
        x = PReLU(shared_axes = [1,2])(x)

    '''
        Layer 4: Expanding
    '''
    x = Conv2D(d,(1,1),padding='same')(x)
    x = PReLU(shared_axes =[1,2])(x)

    '''
        Layer 5
        Deconvolution (upsampling)

    '''
    x= Conv2DTranspose(channels,(9,9),strides=2,padding='same')(x)
    return Model(inputs=inputs,outputs=x)


def train_model(model, lr_images, hr_images, batch_size=16, epochs=50):
    '''
        SGD optimizer is used with learning_rate = 0.00001, momentum=0.9
        the cost function is mean squared error
    '''
    # sgd = SGD(learning_rate=0.0001, momentum=0.9)
    sgd = SGD(learning_rate=0.001, momentum=0.9)

    model.compile(optimizer=sgd, loss='mse')

    model.fit(lr_images, hr_images, batch_size=batch_size, epochs=epochs, verbose=1)

    # model.save('fsrcnn_model.h5')
    model.save('fsrcnn_model.keras')

def load_dataset(image_dir, scale_factor=2,size=(128,128)):
    lr_images = []
    hr_images = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        target_hr_size = (size[0] * scale_factor, size[1] * scale_factor)
        hr_image = load_img(img_path, color_mode='rgb',target_size=target_hr_size)
        # hr_image = load_img(img_path, color_mode='grayscale',target_size=target_hr_size)
        hr_image = img_to_array(hr_image) / 255.0 # to get the range [0,1]

        lr_image = tf.image.resize(hr_image,[size[0] , size[1]] , method=tf.image.ResizeMethod.BICUBIC)
        # lr_image = lr_image.numpy()

        hr_images.append(hr_image)
        lr_images.append(lr_image)
        print(f"Loaded HR image shape: {hr_image.shape}, LR image shape: {lr_image.shape}")
    return np.array(lr_images), np.array(hr_images)

path = r'E:\semester2\research\basics\datasets\BSDS300-images\BSDS300\images\train'
lr_images, hr_images = load_dataset(path)

# Visualize a sample
plt.subplot(1, 2, 1)
plt.title("Low-Resolution Image")
plt.imshow(lr_images[0])

plt.subplot(1, 2, 2)
plt.title("High-Resolution Image")
plt.imshow(hr_images[0])

plt.show()


def upscale_image(model, lr_image_path, scale_factor=2):
    # lr_image = load_img(lr_image_path, color_mode='grayscale')
    lr_image = load_img(lr_image_path, color_mode='rgb')
    lr_image = img_to_array(lr_image) / 255.0
    lr_image = np.expand_dims(lr_image, axis=0)

    sr_image = model.predict(lr_image)
    print("Shape of the output image:", sr_image.shape)
    print(f"Layer 1: {sr_image[0,0:10,0:10,0]}")
    print(f"Layer 2: {sr_image[0,0:10,0:10,1]}")
    print(f"Layer 3: {sr_image[0,0:10,0:10,2]}")
    sr_image = np.squeeze(sr_image, axis=0)
    print("Min pixel value:", np.min(sr_image))
    print("Max pixel value:", np.max(sr_image))
    sr_image = np.clip(sr_image, 0, 1)
    sr_image = (sr_image * 255).astype(np.uint8)
    plt.imshow(sr_image)
    # plt.imshow(sr_image, cmap='gray')
    plt.title("Upscaled Image")
    plt.axis("off")
    plt.show()

    return sr_image

def main():
    # image_dir = r'E:\semester2\research\basics\BSD100\image_SRF_2'
    # image_dir = r'E:\semester2\research\basics\datasets\BSDS300-images\BSDS300\images\train'
    image_dir = r'E:\semester2\research\basics\datasets\Set14\image_SRF_2'
    scale_factor = 2
    batch_size = 16
    # epochs = 50
    epochs =10
    model_path = 'fsrcnn_model.keras'
    if os.path.exists(model_path):
        # Load the existing model
        model = load_model(model_path)
        print("Loaded existing model from", model_path)
    else:
        lr_images, hr_images = load_dataset(image_dir, scale_factor)

        model = fsrcnn()

        train_model(model, lr_images, hr_images, batch_size=batch_size, epochs=epochs)
    # lr_images, hr_images = load_dataset(image_dir, scale_factor)

    # model = fsrcnn()
    # model.summary()
    # train_model(model, lr_images, hr_images, batch_size=batch_size, epochs=epochs)


    test_image_path = r'testImages/image3.png'
    sr_image = upscale_image(model, test_image_path)
    cv2.imwrite('fsrcnn_test_image.jpg', (sr_image * 255).astype(np.uint8))

if __name__ == '__main__':
    main()
