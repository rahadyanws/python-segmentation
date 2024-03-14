import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import clear_output

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# ----------------------------------------------------------------------------
TRAIN_LENGTH = info.splits['train'].num_examples

# Batch size is the number of examples used in one training example.
# It is mostly a power of 2
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# For VGG16 this is the input size
width, height = 224, 224

# ----------------------------------------------------------------------------
def normalize(input_image, input_mask):
   
    # Normalize the pixel range values between [0:1]
    img = tf.cast(input_image, dtype=tf.float32) / 255.0
    input_mask -= 1
    return img, input_mask
 
@tf.function
def load_train_ds(dataset):
    img = tf.image.resize(dataset['image'], 
                          size=(width, height))
    mask = tf.image.resize(dataset['segmentation_mask'],
                           size=(width, height))
 
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
 
    img, mask = normalize(img, mask)
    return img, mask
 
@tf.function
def load_test_ds(dataset):
    img = tf.image.resize(dataset['image'], 
                          size=(width, height))
    mask = tf.image.resize(dataset['segmentation_mask'], 
                           size=(width, height))
 
    img, mask = normalize(img, mask)
    return img, mask

# ----------------------------------------------------------------------------
train = dataset['train'].map(
	load_train_ds, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_test_ds)

train_ds = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test.batch(BATCH_SIZE)

# ----------------------------------------------------------------------------
def display_images(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 
             'Predicted Mask']
 
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
 
    plt.show()
 
 
for img, mask in train.take(1):
    sample_image, sample_mask = img, mask
    display_list = sample_image, sample_mask
 
display_images(display_list)