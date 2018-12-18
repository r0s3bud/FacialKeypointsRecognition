import numpy as np
import cv2
import face_recognition
from keras.models import load_model
from skimage.transform import resize
import tkinter as tk
from tkinter import filedialog
import uuid
import matplotlib.pyplot as plt
from keras import backend as K
from keras.metrics import binary_crossentropy
import tensorflow as tf
from sklearn.metrics import jaccard_similarity_score
from numpy import dot
from numpy.linalg import norm

def mse(true, pred):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((true.astype("float") - pred.astype("float")) ** 2)
    err /= float(len(true))

def accuracy(true, pred):
    """(TP + TN) / (TP + TN + FP + FN)"""

    true_f = true.flatten()
    pred_f = pred.flatten()

    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    for index, value in enumerate(true_f):
        if(value == 1 and pred_f[index] == 1):
            truePositive = truePositive + 1
        if(value == 0 and pred_f[index] == 0):
            trueNegative = trueNegative + 1
        if(value == 0 and pred_f[index] == 1):
            falsePositive = falsePositive + 1
        if (value == 1 and pred_f[index] == 0):
            falseNegative = falseNegative + 1

    return (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)


    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    int = y_true_f * y_pred_f
    intersection = sum(int[0:len(int)])
    return (2. * intersection + smooth) / (sum(y_true_f[0:len(y_true_f)]) + sum(y_pred_f[0:len(y_pred_f)]) + smooth)

def prepareWindow():
    root = tk.Tk()
    root.withdraw()
    return root

def getInputDimensions():
    return 224,224

def getFrameCount(videoStream):
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    return int(cv2.VideoCapture.get(videoStream, property_id))

def getDimensionsOfStream(videoStream):
    return int(videoStream.get(3)), int(videoStream.get(4))

def createOutputFileStream(frameWidth, frameHeight, frameCount):
    if frameCount == 1:
        guid = str(uuid.uuid4())
        return cv2.VideoWriter('output/images/' + guid + '.jpg', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 16, (frameWidth, frameHeight))
    else:
        exit(-1)

def segmentFacesInStream(videoStream, outStream, faceHeight, faceWidth, facePixels):
    while (videoStream.isOpened()):
        ret, frame = videoStream.read()

        if ret == True:

            final_prediction_mask = np.zeros((frameWidth,frameHeight), dtype=np.int32)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_locations = face_recognition.face_locations(frame)

            for faceIndex, locations in enumerate(face_locations):
            # for i in range(1):
                top, right, bottom, left = locations;
                # top,right,bottom,left = 0,369,369,0

                face_img = frame[top:bottom, left:right, :]
                face_img_shape = ((right - left), (bottom - top))
                face_img = resize(face_img, (faceHeight, faceWidth), mode='constant', preserve_range=True)
                facePixels[0] = face_img
                dummyImage = np.zeros((frameWidth, frameHeight), dtype=np.uint8)

                downsampled_to_actual_horizontal_ratio = faceHeight / face_img_shape[0]
                downsampled_to_actual_vertical_ratio = faceWidth / face_img_shape[1]

                predictions = model.predict(facePixels, verbose=1)
                predictions_over_threshold = (predictions > 0.5).astype(np.uint8)

                predictions_upsampled = np.zeros((1, face_img_shape[0], face_img_shape[1], 1),
                                                                dtype=np.uint8)

                predictions_upsampled = resize(predictions,
                                                              (1, face_img_shape[0], face_img_shape[1], 1),
                                                              mode='constant', preserve_range=True)

                predictions_over_threshold_upsampled = np.zeros((1, face_img_shape[0], face_img_shape[1], 1),
                                                                dtype=np.uint8)

                predictions_over_threshold_upsampled = resize(predictions_over_threshold,
                                                              (1, face_img_shape[0], face_img_shape[1], 1),
                                                              mode='constant', preserve_range=True)

                dummyImage = np.zeros((1, frameWidth, frameHeight, 1), dtype=np.uint8)

                prediction_mask = np.zeros((frameWidth, frameHeight), dtype=np.int32)

                prediction_mask_probabilities = np.zeros((frameWidth, frameHeight), dtype=np.int32)

                for dummyIndex, three_dim in enumerate(predictions_over_threshold_upsampled):
                    for rowIndex, row in enumerate(three_dim):
                        for columnIndex, value in enumerate(row):
                            if (value[0] == 1):
                                prediction_mask[columnIndex+left][rowIndex+top] = 1
                                final_prediction_mask[columnIndex+left][rowIndex+top] = 1
                                prediction_mask_probabilities[columnIndex][rowIndex] = predictions_upsampled[0][columnIndex][rowIndex][0]
                                cv2.circle(frame, (columnIndex + left, rowIndex + top), 1, (0, 0, 255), thickness=1,
                                           lineType=8, shift=0)

                prediction_mask_face = prediction_mask[top:bottom, left:right]
                prediction_mask_probabilities_face = prediction_mask_probabilities[top:bottom, left:right]

            cv2.imshow('frame', frame)
            outStream.write(frame)

            filePath2 = filedialog.askopenfilename()

            refImg = cv2.imread(filePath2, 0)

            ref_array = np.zeros((frameWidth, frameHeight), dtype=np.int32)


            for rowIndex, row in enumerate(refImg):
                for columnIndex, value in enumerate(row):
                    if (value > 0):
                        ref_array[columnIndex][rowIndex] = 1

            dsc = dice_coef(ref_array, final_prediction_mask)
            dsc_rounded = round(dsc, 4)
            print('dsc' + str(faceIndex) + ': ' + str(dsc_rounded))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

def closeStreamsAndWindows(videoStream, outStream):
    videoStream.release()
    outStream.release()
    cv2.destroyAllWindows()

window = prepareWindow()
faceHeight, faceWidth = getInputDimensions()
facePixels = np.zeros((1, faceHeight, faceWidth, 3), dtype=np.uint8)

model = load_model('model_post_augmentation.h5', compile = False)

file_path = filedialog.askopenfilename()
videoStream = cv2.VideoCapture(file_path)
frameCount = getFrameCount(videoStream)
frameWidth, frameHeight = getDimensionsOfStream(videoStream)

outStream = createOutputFileStream(frameWidth, frameHeight, frameCount)
segmentFacesInStream(videoStream, outStream, faceHeight, faceWidth, facePixels)
closeStreamsAndWindows(videoStream, outStream)









