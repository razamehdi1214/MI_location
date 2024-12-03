### code for multifidelity training/testing #######
import tensorflow as tf
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
from keras import backend as K
from multifidelity_NN import DNN

###### loading highfidelity human training data ######
from dataset_high_fidelity import x_train_hf,y_train_hf,x_val_hf,y_val_hf

##### loading low fidelity training data ######
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

#### loading low fidelity testing data ######
x_val_lf = np.load("x_test.npy")
y_val_lf = np.load("y_test.npy")

# Converting to TensorFlow tensors
x_train_lf = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_lf = tf.convert_to_tensor(y_train, dtype=tf.float32)

x_val_lf = tf.convert_to_tensor(x_val_lf, dtype=tf.float32)
y_val_lf = tf.convert_to_tensor(y_val_lf, dtype=tf.float32)

x_train_hf = tf.convert_to_tensor(x_train_hf, dtype=tf.float32)
y_train_hf = tf.convert_to_tensor(y_train_hf, dtype=tf.float32)

x_val_hf = tf.convert_to_tensor(x_val_hf, dtype=tf.float32)
y_val_hf = tf.convert_to_tensor(y_val_hf, dtype=tf.float32)


batch_size = 128

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_lf, y_train_lf))
train_dataset = train_dataset.batch(batch_size)

tf.keras.backend.set_floatx('float32')

seed = 42
np.random.seed = seed

model = DNN()
dummy1,dummy2 = model(x_train_lf,x_train_hf)

model.summary()

############################################################
############################################################

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Create a circular mask for the 128x128 image
def create_circular_mask(height, width, radius=63):
    center = (int(height / 2), int(width / 2))
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    mask = dist_from_center <= radius
    return mask.astype(np.float32)

# Generate mask for the input image shape
IMG_HEIGHT, IMG_WIDTH = 128, 128
circular_mask = create_circular_mask(IMG_HEIGHT, IMG_WIDTH)
circular_mask = tf.convert_to_tensor(circular_mask, dtype=tf.float32)

samples = 0  

model.load_weights('saved_models_2_save5/model_best') # Load the model weights

# Make predictions
_, y_val_hf_pred = model(x_val_hf, x_val_hf, train=False)
y_val_hf_pred = (y_val_hf_pred.numpy() > 0.5).astype(int)

# Extract the prediction and ground truth for the circular region
prediction = y_val_hf_pred[samples, :, :, 0]
prediction_masked = prediction * circular_mask

# Load ground truth for the sample
y_val_hf = (y_val_hf.numpy() > 0.5).astype(int)
ground_truth = y_val_hf[samples, :, :, 0]
ground_truth_masked = ground_truth * circular_mask


plt.figure(figsize=(6, 6))  # Adjust the figure size if needed
plt.imshow(prediction_masked)  # Add 'cmap' for grayscale display

# Remove axis
plt.axis('off')

# Save the figure with no extra white space
plt.savefig("plot_pred_sample.png", bbox_inches='tight', pad_inches=0)
plt.show()

# Define functions for calculating metrics
def dice_score(masked_y_true, masked_y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(masked_y_true * masked_y_pred)
    union = tf.reduce_sum(masked_y_true) + tf.reduce_sum(masked_y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def accuracy(masked_y_true, masked_y_pred, circular_mask):
    correct = tf.reduce_sum(tf.cast(masked_y_true == masked_y_pred, tf.float32) * circular_mask)
    total = tf.reduce_sum(circular_mask)
    return correct / total

def precision(masked_y_true, masked_y_pred, smooth=1e-6):
    true_positives = tf.reduce_sum(masked_y_true * masked_y_pred)
    predicted_positives = tf.reduce_sum(masked_y_pred)
    return (true_positives + smooth) / (predicted_positives + smooth)

def recall(masked_y_true, masked_y_pred, smooth=1e-6):
    true_positives = tf.reduce_sum(masked_y_true * masked_y_pred)
    actual_positives = tf.reduce_sum(masked_y_true)
    return (true_positives + smooth) / (actual_positives + smooth)

def iou(masked_y_true, masked_y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(masked_y_true * masked_y_pred)
    union = tf.reduce_sum(masked_y_true) + tf.reduce_sum(masked_y_pred)
    iou = (intersection + smooth) / (union - intersection + smooth)
    return iou


# Calculate and print metrics for each test sample
metrics_results = {
    'dice_infarct': [],
    'accuracy_infarct': [],
    'precision_infarct': [],
    'recall_infarct': [],
    'iou_infarct': []
}

for i in range(len(y_val_hf)):
    # Mask and threshold the ground truth
    y_test_sample = y_val_hf[i, :, :, 0]
    thresholded_y_test = tf.cast(y_test_sample > 0, tf.float32)
    masked_y_test = thresholded_y_test * circular_mask

    # Mask and threshold the prediction
    y_pred_sample = y_val_hf_pred[i, :, :, 0]
    thresholded_y_pred = tf.cast(y_pred_sample > 0, tf.float32)
    masked_y_pred = thresholded_y_pred * circular_mask

    # Calculate metrics for infarct region
    dice_infarct = dice_score(masked_y_test, masked_y_pred)
    acc_infarct = accuracy(masked_y_test, masked_y_pred, circular_mask).numpy()
    prec_infarct = precision(masked_y_test, masked_y_pred).numpy()
    rec_infarct = recall(masked_y_test, masked_y_pred).numpy()
    iou_infarct = iou(masked_y_test, masked_y_pred).numpy()

    metrics_results['dice_infarct'].append(dice_infarct.numpy())
    metrics_results['accuracy_infarct'].append(acc_infarct)
    metrics_results['precision_infarct'].append(prec_infarct)
    metrics_results['recall_infarct'].append(rec_infarct)
    metrics_results['iou_infarct'].append(iou_infarct)


    # Print metrics for each sample
    print(f"Sample {i}:")
    print(f" - Dice Score (Infarct): {dice_infarct.numpy()}")
    print(f" - Accuracy (Infarct): {acc_infarct}")
    print(f" - Precision (Infarct): {prec_infarct}")
    print(f" - Recall (Infarct): {rec_infarct}")
    print(f" - IoU (Infarct): {iou_infarct}")

# Calculate average metrics across all test samples
average_metrics = {metric: np.mean(scores) for metric, scores in metrics_results.items()}
print("\nAverage Metrics Across All Test Samples:")
for metric, avg_score in average_metrics.items():
    print(f" - {metric.capitalize()}: {avg_score}")