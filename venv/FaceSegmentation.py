from keras.models import Model, load_model
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import uuid
import cv2

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_CHANNELS = 3

root = tk.Tk()
root.withdraw()

model = load_model('model_best.h5', custom_objects={'mean_iou': mean_iou}, compile = False)

X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

# fileNames = next(os.walk('testing/'))[2]
#
# for fileName in fileNames:

file_path = filedialog.askopenfilename()

guid = str(uuid.uuid4())
outStream = cv2.VideoWriter('output/images/' + guid + '.jpg', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 16, (369, 369))

img = imread(file_path)[:,:,:IMG_CHANNELS]
img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
X_test[0] = img

preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

imageWithKeypoints = X_test[0]
keypoints = preds_test_t[0][:,:,0]

for rowIndex, row in enumerate(keypoints):
    for columnIndex, value in enumerate(row):
        # if value > 0 and (keypoints[rowIndex - 1][columnIndex] == 0 or keypoints[rowIndex + 1][columnIndex] == 0) :
        if value > 0:
            plt.scatter(columnIndex, rowIndex, c='red')

fig = plt.imshow(np.squeeze(imageWithKeypoints))
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('output/images/' + guid + '.jpg', bbox_inches='tight', pad_inches=0)