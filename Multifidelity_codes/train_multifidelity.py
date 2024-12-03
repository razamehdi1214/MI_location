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
n = 0
nmax_Adam = 40
lr = 5e-4
############################################################

##########################################################
optimizer = tf.optimizers.Adam(learning_rate=lr)
#optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
train_err_list = []
test_err_list = []
train_loss_list = []
test_loss_list = []
n_list = []


train_loss_list = []
test_loss_list = []
train_dice_lf_list = []
train_dice_hf_list = []
val_dice_lf_list = []
val_dice_hf_list = []


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Define the Dice coefficient metric function with masking
def dice(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def plot(n):
    # Plot training and test dice
    plt.figure(figsize=(10, 5))
    plt.plot(n_list, train_dice_lf_list, label='Train Dice Murine')
    plt.plot(n_list, val_dice_lf_list, label='Test Dice Murine')
    plt.plot(n_list, train_dice_hf_list, label = 'Train Dice Human')
    plt.title('DSC')
    plt.xlabel('Iterations')
    plt.ylabel('DSC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DSC_real.png')
    plt.close('all')

    # Plot training and test dice
    plt.figure(figsize=(10, 5))
    plt.semilogy(n_list, train_dice_lf_list, label='Train Dice Murine')
    plt.semilogy(n_list, val_dice_lf_list, label='Test Dice Murine')
    plt.semilogy(n_list, train_dice_hf_list, label = 'Train Dice Human')
    plt.title('DSC')
    plt.xlabel('Iterations')
    plt.ylabel('DSC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DSC_log.png')
    plt.close('all')

    # Plot training and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(n_list, train_loss_list, label='Train loss')
    plt.plot(n_list, test_loss_list, label='Test loss')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_real.png')
    plt.close('all')


    # Plot training and test loss
    plt.figure(figsize=(10, 5))
    plt.semilogy(n_list, train_loss_list, label='Train loss')
    plt.semilogy(n_list, test_loss_list, label='Test loss')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_log.png')
    plt.close('all')


def loss(tr, pred):
    out = tf.reduce_mean(tf.square(pred - tr))
    return out

weigh = model.trainable_variables

# Initialize variables to track the best Dice score and iteration
best_dice_hf = 0
best_iteration = 0

###### training loop ########
while n <= nmax_Adam:
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:  # Removed persistent=True as it's unnecessary
            tape.watch(weigh)
            y_lf_pred, y_hf_pred = model(x_batch, x_train_hf)
            loss_train = loss(y_batch, y_lf_pred) + loss(y_train_hf, y_hf_pred) + tf.reduce_mean(model.losses)
        
        gradients_fs = tape.gradient(loss_train, weigh)
        optimizer.apply_gradients(zip(gradients_fs, weigh))

    # Compute metrics for the current training batch
    err_train_lf = dice(y_batch, y_lf_pred)
    err_train_hf = dice(y_train_hf, y_hf_pred)
    y_val_lf_pred, _ = model(x_val_lf, x_val_hf, train=False)
    err_val_lf = dice(y_val_lf, y_val_lf_pred)

    ###### Evaluate on validation data and track the best weights ########
    if n % 10 == 0:
        loss_test = loss(y_val_lf, y_val_lf_pred)

        
        # Track the best Dice score for human training data
        if err_train_hf > best_dice_hf:
            best_dice_hf = err_train_hf
            best_iteration = n
            model.save_weights(f'saved_models_best/my_model_best')  # Save best weights
        
        # Print progress
        print(f"Iteration: {n}, Train_loss: {loss_train:.4f}, Train_Dice_LF: {err_train_lf:.4f}, "
              f"Val_Dice_LF: {err_val_lf:.4f}, Train_Dice_HF: {err_train_hf:.4f}")
        
        # Log results for plotting
        n_list.append(n)
        train_loss_list.append(loss_train)
        test_loss_list.append(loss_test)
        train_dice_lf_list.append(err_train_lf)
        val_dice_lf_list.append(err_val_lf)
        train_dice_hf_list.append(err_train_hf)

        plot(n)

    n += 1

# Print the best weights information
print(f"The best Dice score for human training data was {best_dice_hf:.4f} at iteration {best_iteration}.")
print(f"Use the weights saved at 'saved_models_best/my_model_best'.")
