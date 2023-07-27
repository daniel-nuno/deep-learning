#%%
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

import tensorflow as tf
from tensorflow.data import AUTOTUNE

from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

import tqdm
import random
import pathlib
import itertools
import collections

#because my emviroment is crasing when trying to plot
#https://stackoverflow.com/questions/59576397/python-kernel-dies-on-jupyter-notebook-with-tensorflow-2
#https://github.com/openai/spinningup/issues/16
#https://stackoverflow.com/questions/65734044/kernel-appears-to-have-died-jupyter-notebook-python-matplotlib
os.environ['KMP_DUPLICATE_LIB_OK']='True'

IMG_SIZE = 75
MAX_SEQ_LENGTH = 10
NUM_FEATURES = 2048

BATCH_SIZE = 8
EPOCHS = 5

DATASET_SIZE = 676

#%%
def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to resized and padded. 
        output_size: Pixel size of the output frame image.

    Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    #frame = tf.image.resize_with_pad(frame, *output_size)
    frame = tf.image.resize(frame, output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size = (75,75), frame_step = 5):
    """
    Creates frames from each video file present for each category.

    Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

    Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    
    return result

class FrameGenerator:
    def __init__(self, path, n_frames, training = False):
        """ Returns a set of frames with their associated label. 

        Args:
            path: Video file paths.
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            label = self.class_ids_for_name[name] # Encode labels
            yield video_frames, label
#%%
download_dir = pathlib.Path("C:/Users/nuno/Desktop/deep-learning-data/exam2/UFC-5/")
fg = FrameGenerator(download_dir, MAX_SEQ_LENGTH, training=True)

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")

#%%

# Create the training set
output_signature = (tf.TensorSpec(shape = (MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
#dataset = tf.data.Dataset.from_generator(FrameGenerator(download_dir, 10, training=True),
#                                          output_signature = output_signature)

dataset = tf.data.Dataset.from_generator(FrameGenerator(download_dir, MAX_SEQ_LENGTH, training=True), output_types=(tf.float32, tf.int16))

dataset = (dataset
                .cache()
                .shuffle(676)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE)
                )

train_dataset = dataset.take(int(DATASET_SIZE*0.8))
validation_dataset = dataset.take(int(DATASET_SIZE*0.2))

#train_dataset = train_dataset.cache().shuffle(676).prefetch(buffer_size = AUTOTUNE)
#validation_dataset = validation_dataset.cache().shuffle(676).prefetch(buffer_size = AUTOTUNE)
#%%
input_shape = (MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
num_classes = 5

cnn = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
cnn.trainable = False

inputs = Input(shape=input_shape)
x = keras.layers.TimeDistributed(cnn)(inputs)
x = keras.layers.GRU(256, return_sequences=False)(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

rnn = Model(inputs=inputs, outputs=outputs)
rnn.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy())

filepath = "C:/Users/nuno/Desktop/deep-learning-data/exam2/tmp/video_classifier"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1)

#%%
rnn.summary()
#%%

#%%

#history = rnn.fit(x=dataset, epochs=EPOCHS)

history = rnn.fit(x=train_dataset,
                  epochs=EPOCHS,
                  callbacks=[checkpoint],
                  validation_data=validation_dataset
    )

# %%
def plot(var1, var2, plot_name):
    # Get the loss metrics from the trained model
    c1 = history.history[var1]
    c2 = history.history[var2]

    epochs = range(len(c1))

    # Plot the metrics
    plt.plot(epochs, c1, 'b', label=var1)
    plt.plot(epochs, c2, 'r', label=var2)
    plt.title(str(plot_name))
    plt.legend()

plot( 'loss', 'val_loss', 'Training Loss vs Validation Loss')
#%%
plot( 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy', 'Training Accuracy vs Validation Accuracy')