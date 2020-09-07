# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import IPython
import IPython.display as display

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import binary_crossentropy

import cv2



# Train specification
epochs = 5
lr = 0.0001

bunny = np.load('bunny.npy')
bunny = bunny

cow = np.load('cow.npy')
cow = cow
# Normalized between 0-1



num_rows, num_cols = bunny.shape[:2]

# bottom right: 46 39 to 60 70
translation_matrix = np.float32([ [1,0,0], [0,1,0] ])
img_translation = cv2.warpAffine(cow, translation_matrix, (num_cols, num_rows))
plt.imshow(img_translation,cmap='gray')



# Creates train data
biased_train_bun = []
biased_train_cow = []

for i in range(1000):
  
  x_val = np.random.randint(46,60)
  y_val = np.random.randint(39,70)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
  biased_train_bun.append(im)
  x_val = np.random.randint(-53,-39)
  y_val = np.random.randint(-65,-34)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
  biased_train_bun.append(im)
  
  y_val = np.random.randint(44,64)
  x_val = np.random.randint(-46,-22)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(cow, translation_matrix, (200, 200))
  biased_train_cow.append(im)
  y_val = np.random.randint(-56,-33)
  x_val = np.random.randint(55,78)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(cow, translation_matrix, (200, 200))
  biased_train_cow.append(im)



# Creates test data
unbiased_train_bun = []
unbiased_train_cow = []

for i in range(2000):
  
  x_val = np.random.randint(-53,60)
  y_val = np.random.randint(-65,70)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
  unbiased_train_bun.append(im)
  
  y_val = np.random.randint(-56,64)
  x_val = np.random.randint(-46,78)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(cow, translation_matrix, (200, 200))
  unbiased_train_cow.append(im)

biased_train_cow = np.array(biased_train_cow)
biased_train_bun = np.array(biased_train_bun)
unbiased_train_cow = np.array(unbiased_train_cow)
unbiased_train_bun = np.array(unbiased_train_bun)

unbiased_squares = unbiased_train_cow
unbiased_hearts = unbiased_train_bun
biased_squares = biased_train_cow
biased_hearts = biased_train_bun



# Print and Save some examples
some_unbiased_squares = biased_squares[:25]
np.random.shuffle(some_unbiased_squares)

x = 5
y = 5

fig,axarr = plt.subplots(x,y)

for ax,im in zip(axarr.ravel(), np.float32(some_unbiased_squares)):
    ax.axis('off')
    ax.imshow(im,cmap = 'gray')

fig.savefig('biased_stars.png')



# Prepare Training and Testing Data
unbiased_data = np.concatenate((unbiased_squares,unbiased_hearts))
unbiased_data = np.float32(unbiased_data)

biased_data = np.concatenate((biased_squares,biased_hearts))
biased_data = np.float32(biased_data)

labels = np.concatenate((np.zeros(len(unbiased_squares)),np.ones(len(unbiased_squares))))

# For the dense network 
train_images = np.reshape(biased_data,(-1, 200*200)).astype("float32") 
train_dataset = tf.data.Dataset.from_tensor_slices((train_images,labels)).shuffle(200)
train_dataset = train_dataset.batch(100)

test_images = np.reshape(unbiased_data,(-1, 200*200)).astype("float32")
test_dataset = tf.data.Dataset.from_tensor_slices((test_images,labels))
test_dataset = test_dataset.batch(100)

# For the convolutional network 
train_images_cnn = np.reshape(biased_data,(-1, 200,200,1)).astype("float32") 
train_dataset_cnn = tf.data.Dataset.from_tensor_slices((train_images_cnn,labels)).shuffle(200)
train_dataset_cnn = train_dataset_cnn.batch(100)

test_images_cnn = np.reshape(unbiased_data,(-1, 200,200,1)).astype("float32")
test_dataset_cnn = tf.data.Dataset.from_tensor_slices((test_images_cnn,labels))
test_dataset_cnn = test_dataset_cnn.batch(100)



# Build Networks

# Create the dense network
DNN = keras.Sequential([
    keras.Input(shape=200*200, name="dense_input"),
    layers.Dense(units=1024),
    layers.ReLU(),
    layers.Dense(units=1024),
    layers.ReLU(),
    layers.Dense(units=1, activation = "sigmoid")
], name='dense')

# Create the convolutional network v1
CNN = keras.Sequential([
    keras.Input(shape=(200,200,1), name="conv_input"),
    layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same'),
    layers.ReLU(),
    layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same'),
    layers.ReLU(),
    layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same'),
    layers.ReLU(),
    layers.AveragePooling2D(200),
    layers.Flatten(),
    layers.Dense(units=1, activation = "sigmoid")
], name='convnetwork')

# Create the convolutional network v2
CNN2 = keras.Sequential([
    keras.Input(shape=(200,200,1), name="conv2_input"),
    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.ReLU(),
    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.ReLU(),
    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.ReLU(),
    layers.AveragePooling2D(25),
    layers.Flatten(),
    layers.Dense(units=1, activation = "sigmoid")
], name='convnetwork2')



# Compile and Train the dense model
DNN.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

DNN.fit(train_dataset, epochs=epochs)

score = DNN.evaluate(test_dataset, verbose=0)
print("Dense network test loss:", score[0])
print("Dense network test accuracy:", score[1])