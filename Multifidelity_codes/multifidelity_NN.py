################# main code for multifidelity ################
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
import numpy as np
import os
import re
import shutil
import random
from datetime import datetime


class DNN(tf.keras.Model):
    def __init__(self):
        super(DNN, self).__init__()
    
        seed = 6
        np.random.seed = seed

        IMG_WIDTH = 128
        IMG_HEIGHT = 128
        IMG_CHANNELS = 3

        #Build the UNet model
        self.s_lf = tf.keras.layers.Lambda(lambda x: x / 255)

        #Contraction path
        self.c1_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c1_d = tf.keras.layers.Dropout(0.1)
        self.c1_2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.p1 = tf.keras.layers.MaxPooling2D((2, 2))

        self.c2_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c2_d = tf.keras.layers.Dropout(0.1)
        self.c2_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.p2 = tf.keras.layers.MaxPooling2D((2, 2))

        self.c3_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c3_d = tf.keras.layers.Dropout(0.2)
        self.c3_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.p3 = tf.keras.layers.MaxPooling2D((2, 2))

        self.c4_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c4_d = tf.keras.layers.Dropout(0.2)
        self.c4_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.c5_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c5_d = tf.keras.layers.Dropout(0.3)
        self.c5_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        #Expansive path
        self.u6_1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.u6_2 = tf.keras.layers.Concatenate(axis=3)
        self.c6_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c6_2 = tf.keras.layers.Dropout(0.2)
        self.c6_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.u7_1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.u7_2 = tf.keras.layers.Concatenate(axis=3)
        self.c7_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c7_2 = tf.keras.layers.Dropout(0.2)
        self.c7_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.u8_1 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
        self.u8_2 = tf.keras.layers.Concatenate(axis=3)
        self.c8_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c8_2 = tf.keras.layers.Dropout(0.1)
        self.c8_3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.u9_1 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
        self.u9_2 = tf.keras.layers.Concatenate(axis=3)
        self.c9_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.c9_2 = tf.keras.layers.Dropout(0.1)
        self.c9_3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.y_lf = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        self.s_hf = tf.keras.layers.Lambda(lambda x: x / 255)
        self.y_pred_hf_l = tf.keras.layers.Conv2D(1,(10,10),kernel_initializer = 'glorot_normal',padding='same', kernel_regularizer = tf.keras.regularizers.L1(0.01))
        self.y_pred_hf_nl1 = tf.keras.layers.Conv2D(1,(10,10) , kernel_initializer = 'glorot_normal',padding='same', kernel_regularizer = tf.keras.regularizers.L1(0.01))
        # self.y_pred_hf_nl2 = tf.keras.layers.Conv2D(1,(10,10) , kernel_initializer = 'glorot_normal',padding='same', kernel_regularizer = tf.keras.regularizers.L1(0.01))
        # self.y_pred_hf_nl3 = tf.keras.layers.Conv2D(1,(10,10) , kernel_initializer = 'glorot_normal',padding='same', kernel_regularizer = tf.keras.regularizers.L1(0.01))
        self.y_hf = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')

    def forward_lf(self,x_lf,train = True):
        c1 = self.s_lf(x_lf)
        #Contraction path
        c1 = self.c1_1(c1)
        c1 = self.c1_d(c1,training = train)
        c1 = self.c1_2(c1)
        p1 = self.p1(c1)

        c2 = self.c2_1(p1)
        c2 = self.c2_d(c2,training = train)
        c2 = self.c2_2(c2)
        p2 = self.p2(c2)

        c3 = self.c3_1(p2)
        c3 = self.c3_d(c3, training = train)
        c3 = self.c3_2(c3)
        p3 = self.p3(c3)

        c4 = self.c4_1(p3)
        c4 = self.c4_d(c4, training = train)
        c4 = self.c4_2(c4)
        p4 = self.p4(c4)

        c5 = self.c5_1(p4)
        c5 = self.c5_d(c5, training = train)
        c5 = self.c5_2(c5)


        u6 = self.u6_1(c5)
        u6 = self.u6_2([u6, c4])
        u6 = self.c6_1(u6)
        u6 = self.c6_2(u6, training = train)
        u6 = self.c6_3(u6)

        u6 = self.u7_1(u6)
        u6 = self.u7_2([u6, c3])
        u6 = self.c7_1(u6)
        u6 = self.c7_2(u6, training = train)
        u6 = self.c7_3(u6)

        u6 = self.u8_1(u6)
        u6 = self.u8_2([u6, c2])
        u6 = self.c8_1(u6)
        u6 = self.c8_2(u6, training = train)
        u6 = self.c8_3(u6)


        u6 = self.u9_1(u6)
        u6 = self.u9_2([u6, c1])
        u6 = self.c9_1(u6)
        u6 = self.c9_2(u6, training = train)
        u6 = self.c9_3(u6)
        
        u6 = self.y_lf(u6)

        return u6

    def forward_hf(self,x_hf,train = True):
        y_pred_hf_l = self.y_pred_hf_l(x_hf)
        y_pred_hf_nl = tf.nn.leaky_relu(self.y_pred_hf_nl1(x_hf))
        # y_pred_hf_nl = tf.nn.leaky_relu(self.y_pred_hf_nl2(y_pred_hf_nl))
        # y_pred_hf_nl = tf.nn.leaky_relu(self.y_pred_hf_nl3(y_pred_hf_nl))
        y_pred_hf = y_pred_hf_nl + y_pred_hf_l
        y_pred_mult = self.y_hf(y_pred_hf)
        
        return y_pred_mult


    @tf.function
    def call(self, x_lf,x_hf, train = True):

        x_lf = self.s_lf(x_lf)
        x_hf = self.s_hf(x_hf)
        threshold = 0.5

        y_lf = self.forward_lf(x_lf,train = train)
        y_lf_hf = self.forward_lf(x_hf,train = train)

        temp = tf.concat([x_hf,y_lf_hf],axis = -1)
        y_hf = self.forward_hf(temp,train = train)

        y_hf = tf.where(y_hf > threshold, y_hf, tf.constant(threshold, dtype=y_hf.dtype))

        return y_lf,y_hf

