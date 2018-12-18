import face_recognition
import matplotlib.patches as ptc
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import os

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

image = face_recognition.load_image_file("testing/Ola.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

model = load_model('model-dsbowl2018-3.h5', custom_objects={'mean_iou': mean_iou}, compile = False)

X_test = np.zeros((1, 96, 96, 3), dtype=np.uint8)

for (top, right, bottom, left) in  face_locations:
    # # Create figure and axes
#     # fig, ax = plt.subplots(1)
#     #
#     # # Display the image
#     # ax.imshow(image, cmap="gray")
#     #
#     # rectangle1 = ptc.Rectangle((left, top), (right - left), (bottom - top), linewidth=1, edgecolor='r',facecolor='none')
#     # ax.add_patch(rectangle1)
#     # plt.show()

    img = image[top:bottom,left:right,:]
    img = resize(img, (96, 96), mode='constant', preserve_range=True)
    X_test[0] = img

    preds_test = model.predict(X_test, verbose=1)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(X_test[0], cmap="gray")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(np.squeeze(preds_test_t[0]))
    ax3 = fig.add_subplot(2, 2, 3)
    imageWithKeypoints = X_test[0]
    keypoints = preds_test_t[0][:, :, 0]
    ax3.imshow(np.squeeze(imageWithKeypoints))
    for rowIndex, row in enumerate(keypoints):
        for columnIndex, value in enumerate(row):
            if value > 0:
                ax3.scatter(columnIndex, rowIndex, c='red')

    plt.show()