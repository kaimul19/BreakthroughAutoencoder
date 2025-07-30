from astropy import units as u
import numpy as np
import setigen as stg
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
import random


def generate_welsh_flag_array(location = 'WelshFlag.npy'):
    # Load the image
    image = np.load(location)

    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    buffer = np.zeros((1000,1000))

    buffer_height, buffer_width = buffer.shape
    image_height, image_width = normalized_image.shape

    # Compute the start indices
    start_y = (buffer_height - image_height) // 2
    start_x = (buffer_width - image_width) // 2

    # Add the image to the buffer
    buffer[start_y:start_y + image_height, start_x:start_x + image_width] = normalized_image

    # if random.choice([-1, 1]) == 1:
    #     buffer = cv2.flip(buffer, 1)

    # if random.choice([-1, 1]) == 1:
    #     buffer = cv2.flip(buffer, 0)

    # if random.choice([-1, 1]) == 1:
    #     buffer = cv2.transpose(buffer)

    # if random.choice([-2, -1, 1, 2]) != 1:
    #     scale = 1
    #     angle = random.uniform(0,359)
    #     rotation_matrix = cv2.getRotationMatrix2D(((len(buffer[:,0])//2,len(buffer[0,:])//2)), angle, scale)
        
    #     # Perform the rotation
    #     buffer = cv2.warpAffine(buffer, rotation_matrix, (1000, 1000))

    non_zero_indices = np.argwhere(buffer > 0)
    top_left = non_zero_indices.min(axis=0)
    bottom_right = non_zero_indices.max(axis=0)

    # Crop the image
    buffer = buffer[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

    # plt.axis('off')
    # plt.imshow(buffer, cmap='gray')
    # plt.show()

    m = int(random.uniform(10, 16))
    n = int(m*1.6*random.uniform(1,2.5))*20  # Change n to your desired dimension

    # m = 10
    # n= int(m*192/10)
    # Resize the cropped image to nxn while maintaining aspect ratio
    # final resize – switch to nearest-neighbour
    resized_image = cv2.resize(buffer, (n, m), interpolation=cv2.INTER_NEAREST)

    return resized_image


