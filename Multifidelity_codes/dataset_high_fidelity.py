################# Code for loading the Human Scans & Data Augmentation##############################

import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import re
from PIL import Image
import shutil
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

def load_images(input_folder, desired_height, desired_width):
    input_images = []
    loaded_indices = []

    file_pattern = re.compile(r"(circum|radial|longit)_(\d+)\.png")

    image_paths = {}

    for file in os.listdir(input_folder):
        match = file_pattern.match(file)
        if match:
            category, index = match.groups()
            index = int(index)

            if index not in image_paths:
                image_paths[index] = {}

            image_paths[index][category] = os.path.join(input_folder, file)

    for index in sorted(image_paths.keys()):
        paths = image_paths[index]
        if all(category in paths for category in ["circum", "radial", "longit"]):
            images = [Image.open(paths[category]).convert("L").resize((desired_width, desired_height))
                      for category in ["circum", "radial", "longit"]]

            combined_image = np.stack(images, axis=-1)
            input_images.append(combined_image)
            loaded_indices.append(index)  # Save the index

    return np.array(input_images), loaded_indices

# Load images and get their indices
input_folder = 'Human_CMR/' # locate data folder for inputs
desired_height = 128
desired_width = 128

input_images, indices = load_images(input_folder, desired_height, desired_width)

# Verify the shape of the input_images array
print(f"Loaded {input_images.shape[0]} sets of images.")
print(f"Indices of loaded images: {indices}")

output_folder = 'Human_CMR/'  # locate data folder for outputs
output_images = []

# Resize each image to 128x128, add a channel dimension, and display it
for i in indices:
    output_paths = [output_folder + f"patient_{i}_LGE.png"]
    images = []

    for path in output_paths:
        img = Image.open(path).convert("L").resize((128, 128))  # Resize to 128x128
        img_array = np.array(img)[..., np.newaxis]  # Add channel dimension (H, W) -> (H, W, 1)
        images.append(img_array)

    output_images.extend(images)  # Append images to the output list

# Convert the list to a NumPy array with shape (5, 128, 128, 1)
output_images = np.array(output_images)


threshold_value = 100
output_images[output_images <= threshold_value] = 0
output_images[output_images > threshold_value] = 255


# Visualize each image
for i in range(5):
    plt.imshow(input_images[i])
    plt.title(f"Image {i}")
    plt.savefig("Input_image" + str(i) + ".png")
    plt.close()

# Visualize each image
for i in range(5):
    plt.imshow(output_images[i])
    plt.title(f"Image {i}")
    plt.savefig("Output_image" + str(i) + ".png")
    plt.close()

# Desired dimensions for resizing
desired_height = 128
desired_width = 128

# Resize the input_images
resized_inputs = np.zeros((input_images.shape[0], desired_height, desired_width, 3), dtype=np.uint8)
for i in range(input_images.shape[0]):
    resized_inputs[i] = resize(input_images[i], (desired_height, desired_width), preserve_range=True, anti_aliasing=True)

# Resize the output_images
resized_outputs = np.zeros((output_images.shape[0], desired_height, desired_width, 1), dtype=bool)
for i in range(output_images.shape[0]):
    resized_outputs[i] = resize(output_images[i], (desired_height, desired_width), preserve_range=True, anti_aliasing=True)

     

# Verify the shapes of the resized arrays
print('resized_inputs',resized_inputs.shape)    # (..., 128, 128, 3)
print('resized_outputs',resized_outputs.shape)   # (..., 128, 128, 1)

x_train_1 = np.concatenate([resized_inputs[:3,:,:,:], resized_inputs[4:,:,:,:]], axis=0) # to keep out samples other than first
y_train_1 = np.concatenate([resized_outputs[:3,:,:,:], resized_outputs[4:,:,:,:]], axis=0)

# x_train_1 = np.concatenate([resized_inputs[1:,:,:,:]], axis=0) # to keep out first sample
# y_train_1 = np.concatenate([resized_outputs[1:,:,:,:]], axis=0)


x_val_hf = np.expand_dims(resized_inputs[3, :, :, :], axis=0)  
y_val_hf = np.expand_dims(resized_outputs[3, :, :, :], axis=0)  


# Data Augmentation

rotated_images = []

for image in x_train_1:
    rotated_90 = np.rot90(image, k=1)
    rotated_180 = np.rot90(image, k=2)
    rotated_270 = np.rot90(image, k=3)
    rotated_images.extend([rotated_90, rotated_180, rotated_270])

rotated_images = np.array(rotated_images)
rotated_mask = []

for image in y_train_1:
    rotated_90 = np.rot90(image, k=1)
    rotated_180 = np.rot90(image, k=2)
    rotated_270 = np.rot90(image, k=3)
    rotated_mask.extend([rotated_90, rotated_180, rotated_270])

rotated_mask = np.array(rotated_mask)

x_train_hf = np.concatenate((x_train_1, rotated_images), axis=0)
y_train_hf = np.concatenate((y_train_1, rotated_mask), axis=0)
print(x_train_hf.shape)
print(y_train_hf.shape)
print(x_val_hf.shape)
print(y_val_hf.shape)
