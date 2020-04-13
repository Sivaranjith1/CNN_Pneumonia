#%%
import os
import numpy as np
import cv2
import random
import pickle

# %%
DATADIR = "D:/ml/pneumonia/data"
CATEGORIES = ['NORMAL', 'PNEUMONIA']
TRAINING = ['train', 'val']
TESTING = ['test']

# %%
IMG_SIZE = 60

# %%
def create_data(folders):
    output = []
    for folder in folders:
        for category in CATEGORIES:
            path = os.path.join(DATADIR,folder, category)
            class_num = CATEGORIES.index(category)

            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                    output.append([new_array, class_num])
                except Exception as e:
                    pass
    random.shuffle(output)
    return output

# %%
training_data = create_data(TRAINING)
#%%
print(len(training_data))
# %%
def save_training_data(data):
    X = []
    Y = []

    for features, label in training_data:
        X.append(features)
        Y.append(label)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pickle", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()

# %%
save_training_data(training_data)

# %%
test_data = create_data(TESTING)

# %%
len(test_data)

# %%

def save_testing_data(data):
    X = []
    Y = []

    for features, label in data:
        X.append(features)
        Y.append(label)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y_test.pickle", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()

# %%
save_testing_data(test_data)

# %%
