
#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SpatialDropout2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pickle
import time
import os
import numpy as np
#%%
X = np.array(pickle.load(open("X.pickle", "rb")))
Y = np.array(pickle.load(open("Y.pickle", "rb")))

X_test = np.array(pickle.load(open("X_test.pickle", "rb")))
Y_test = np.array(pickle.load(open("Y_test.pickle", "rb")))

X_test = X_test/255.0
X = X/255.0
#%%
LAYERS = [2,3]
FILTERS = [32,64,128]
#%%
# for filt in FILTERS:
#     for layer in LAYERS:
#         NAME = str(filt)+"x"+str(layer)+"-Pneumonia-cnn"
#         tensorboard = TensorBoard(log_dir=(os.path.dirname(os.path.abspath(__file__)) + '\\logs\\two\\'+NAME))
#         checkpoint = ModelCheckpoint(
#             os.path.dirname(os.path.abspath(__file__)) + '\\logs\\checkpoint-2\\'+NAME,
#             monitor='val_loss',
#             verbose=1,
#             save_best_only=True,
#             mode='min'
#             )

#         model = Sequential()
#         for i in range(layer):
#             model.add(Conv2D(filt, (3, 3), input_shape=X.shape[1:]))
#             model.add(Activation("relu"))
#             model.add(SpatialDropout2D(0.7))
#             model.add(MaxPooling2D(pool_size=(3, 3)))
#             model.add(Dropout(0.5))

#         # model.add(Dropout(0.2))
#         model.add(Flatten())
#         model.add(Dense(filt))
#         model.add(Activation('relu'))

#         model.add(Dense(1))
#         model.add(Activation("sigmoid"))

#         model.compile(
#             loss="binary_crossentropy",
#             optimizer="adam",
#             metrics=['accuracy']
#         )


#         model.fit(X, Y, batch_size=64, 
#                 validation_data=(X_test, Y_test),
#                 epochs=40, 
#                 callbacks=[tensorboard, checkpoint])

#         model.save('models/'+NAME + '.model')
# print('Finish')

# %%
layer1 = 64
layer2 = 32
spatial = 0
drop = 0.5
NAME = f'{layer1}-{layer2}-Pneumonia-cnn-{spatial}-{drop}'
print(NAME)
tensorboard = TensorBoard(log_dir=(os.path.dirname(os.path.abspath(__file__)) + '\\logs\\log-5\\'+NAME))
checkpoint = ModelCheckpoint(
    os.path.dirname(os.path.abspath(__file__)) + '\\logs\\checkpoint-5\\'+NAME+'.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
    )

model = Sequential()
model.add(Conv2D(layer1, (2, 2), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(SpatialDropout2D(spatial))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(drop))


model.add(Conv2D(layer2, (3, 3)))
model.add(Activation("relu"))
model.add(SpatialDropout2D(spatial))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(drop))

# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(layer2))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

epoch = 100
to_run = 500
model.fit(X, Y, batch_size=64, 
        validation_data=(X_test, Y_test),
        initial_epoch=epoch,
        epochs=epoch+to_run, 
        callbacks=[tensorboard, checkpoint])

model.save('models/'+NAME + '.model')

# %%
