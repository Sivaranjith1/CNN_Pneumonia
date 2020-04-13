#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import os
import numpy as np

print(os.path.dirname(os.path.abspath(__file__)) + '\\logs')

NAME = "32-64-128-Pneumonia-cnn-evaluate"
tensorboard = TensorBoard(log_dir=(os.path.dirname(os.path.abspath(__file__)) + '\\logs\\'+NAME))

#%%
X = np.array(pickle.load(open("X_test.pickle", "rb")))
Y = np.array(pickle.load(open("Y_test.pickle", "rb")))


X = X/255.0

# %%
model = tf.keras.models.load_model('64-32-Pneumonia-cnn.h5')

# %%
model.summary()

# %%
model.evaluate(X, Y, 
    batch_size=64, callbacks=[tensorboard])

# %%
model.save("64-32-Pneumonia-cnn.h5")

# %%
