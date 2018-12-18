import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, os, warnings, cv2

from keras.models import *
from keras.layers import *


import pandas as pd

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)

FTRAIN = "data/training.csv"
FTEST  = "data/test.csv"
FIdLookup = 'data/IdLookupTable.csv'


def gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)  ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  ## (height,1)
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def generate_hm(height, width, landmarks, s=3):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
    """
    Nlandmarks = len(landmarks)
    hm = np.zeros((height, width, Nlandmarks), dtype=np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1, -1]):

            hm[:, :, i] = gaussian_k(landmarks[i][0],
                                     landmarks[i][1],
                                     s, height, width)
        else:
            hm[:, :, i] = np.zeros((height, width))
    return hm


def get_y_as_heatmap(df, height, width, sigma):
    columns_lmxy = df.columns[:-1]  ## the last column contains Image
    columns_lm = []
    for c in columns_lmxy:
        c = c[:-2]
        if c not in columns_lm:
            columns_lm.extend([c])

    y_train = []
    for i in range(df.shape[0]):
        landmarks = []
        for colnm in columns_lm:
            x = df[colnm + "_x"].iloc[i]
            y = df[colnm + "_y"].iloc[i]
            if np.isnan(x) or np.isnan(y):
                x, y = -1, -1
            landmarks.append([x, y])

        y_train.append(generate_hm(height, width, landmarks, sigma))
    y_train = np.array(y_train)

    return (y_train, df[columns_lmxy], columns_lmxy)


def load(test=False, width=96, height=96, sigma=5):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are
           extracted. for example, cols could be:

          [left_eye_center_x, left_eye_center_y]

    return:
    X:  2-d numpy array (Nsample, Ncol*Nrow)
    y:  2-d numpy array (Nsample, Nlandmarks*2)
        In total there are 15 landmarks.
        As x and y coordinates are recorded, u.shape = (Nsample,30)
    y0: panda dataframe containins the landmarks

    """
    from sklearn.utils import shuffle

    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    myprint = df.count()
    myprint = myprint.reset_index()
    print(myprint)
    ## row with at least one NA columns are removed!
    df = df.dropna()
    df = df.fillna(-1)

    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)

    if not test:  # labels only exists for the training data
        y, y0, nm_landmark = get_y_as_heatmap(df, height, width, sigma)
        X, y, y0 = shuffle(X, y, y0, random_state=42)  # shuffle data
        y = y.astype(np.float32)
    else:
        y, y0, nm_landmark = None, None, None

    return X, y, y0, nm_landmark


def load2d(test=False, width=96, height=96, sigma=5):
    re = load(test, width, height, sigma)
    X = re[0].reshape(-1, width, height, 1)
    y, y0, nm_landmarks = re[1:]

    return X, y, y0, nm_landmarks

sigma = 5

X_train, y_train, y_train0, nm_landmarks = load2d(test=False,sigma=sigma)
X_test,  y_test, _, _ = load2d(test=True,sigma=sigma)
print (X_train.shape,y_train.shape, y_train0.shape)
print (X_test.shape,y_test)

Nplot = y_train.shape[3]+1

for i in range(10):
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(2, Nplot / 2, 1)
    ax.imshow(X_train[i, :, :, 0], cmap="gray")
    ax.set_title("input")
    for j, lab in enumerate(nm_landmarks[::2]):
        ax = fig.add_subplot(2, Nplot / 2, j + 2)
        ax.imshow(y_train[i, :, :, j], cmap="gray")
        ax.set_title(str(j) + "\n" + lab[:-2], fontsize=10)
    plt.show()


