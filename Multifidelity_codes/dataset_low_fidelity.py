######################## Script for loading the RCCM image data & Data Augmentation ##############################
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

input_folder = '/training_data/' # locate data folder for inputs
desired_height = 656
desired_width = 875

input_images, indices = load_images(input_folder, desired_height, desired_width)

output_folder = '/labels/'  # locate data folder for outputs
output_images = []

for i in indices:

      output_paths = [output_folder + f"labels_{i}.png"]

      output_images.append([np.array(Image.open(path).convert("L")) for path in output_paths])

output_images = np.array(output_images)
output_images = np.transpose(output_images, (0,2, 3, 1))  # Transpose the dimensions

threshold_value = 100
output_images[output_images <= threshold_value] = 0
output_images[output_images > threshold_value] = 255

import matplotlib.pyplot as plt

########### Visualize each image #####################
for i in range(5):
    plt.imshow(input_images[i])
    plt.title(f"Image {i}")
    plt.savefig("Input_image" + str(i) + ".png")
    plt.close()

for i in range(5):
    plt.imshow(output_images[i])
    plt.title(f"Image {i}")
    plt.savefig("Input_image" + str(i) + ".png")
    plt.close()

######### Resizing the output #########################

desired_height = 128
desired_width = 128

resized_inputs = np.zeros((input_images.shape[0], desired_height, desired_width, 3), dtype=np.uint8)
for i in range(input_images.shape[0]):
    resized_inputs[i] = resize(input_images[i], (desired_height, desired_width), preserve_range=True, anti_aliasing=True)

resized_outputs = np.zeros((output_images.shape[0], desired_height, desired_width, 1), dtype=bool)
for i in range(output_images.shape[0]):
    resized_outputs[i] = resize(output_images[i], (desired_height, desired_width), preserve_range=True, anti_aliasing=True)

print(resized_inputs.shape)    # (..., 128, 128, 3)
print(resized_outputs.shape)   # (..., 128, 128, 1)

############### Splitting the train and test #################################
x_train_1, x_test, y_train_1, y_test = train_test_split(resized_inputs, resized_outputs, test_size=0.2, random_state=42)


################  Data Augmentation for training data by rotation 90,180,270 ####################
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

x_train_lf = np.concatenate((x_train_1, rotated_images), axis=0)
y_train_lf = np.concatenate((y_train_1, rotated_mask), axis=0)

################# Saving the data #######################
np.save("x_train",x_train_lf)
np.save("y_train",y_train_lf)

np.save("x_test",x_test)
np.save("y_test",y_test)
