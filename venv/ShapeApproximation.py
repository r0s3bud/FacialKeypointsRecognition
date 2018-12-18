import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, os, warnings, cv2, concurrent.futures, multiprocessing
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

from keras.models import *
from keras.layers import *

import pandas as pd

def load2d(test=False, width=96, height=96, sigma=5):
    re, df = load(test, width, height, sigma)
    X = re[0].reshape(-1, width, height, 1)

    return re, df

def load(test=False, width=96, height=96, sigma=5):
    from sklearn.utils import shuffle

    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    #row with at least one NA columns are removed!
    df = df.dropna()
    df = df.fillna(-1)

    myprint = df.iloc(0).obj
    print(myprint)

    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)

    return X, df


def save_images_and_masks(item, i, dfRow):

    if i < 1542:
        return;

    leftEyeCenterX = dfRow['left_eye_center_x']
    leftEyeCenterY = dfRow['left_eye_center_y']
    leftEyeInnerCornerX = dfRow['left_eye_inner_corner_x']
    leftEyeInnerCornerY = dfRow['left_eye_inner_corner_y']
    leftEyeOuterCornerX = dfRow['left_eye_outer_corner_x']
    leftEyeOuterCornerY = dfRow['left_eye_outer_corner_y']

    rightEyeCenterX = dfRow['right_eye_center_x']
    rightEyeCenterY = dfRow['right_eye_center_y']
    rightEyeInnerCornerX = dfRow['right_eye_inner_corner_x']
    rightEyeInnerCornerY = dfRow['right_eye_inner_corner_y']
    rightEyeOuterCornerX = dfRow['right_eye_outer_corner_x']
    rightEyeOuterCornerY = dfRow['right_eye_outer_corner_y']

    leftEyebrowInnerEndX = dfRow['left_eyebrow_inner_end_x']
    leftEyebrowInnerEndY = dfRow['left_eyebrow_inner_end_y']
    leftEyebrowOuterEndX = dfRow['left_eyebrow_outer_end_x']
    leftEyebrowOuterEndY = dfRow['left_eyebrow_outer_end_y']

    rightEyebrowInnerEndX = dfRow['right_eyebrow_inner_end_x']
    rightEyebrowInnerEndY = dfRow['right_eyebrow_inner_end_y']
    rightEyebrowOuterEndX = dfRow['right_eyebrow_outer_end_x']
    rightEyebrowOuterEndY = dfRow['right_eyebrow_outer_end_y']

    mouthLeftCornerX = dfRow['mouth_left_corner_x']
    mouthLeftCornerY = dfRow['mouth_left_corner_y']
    mouthRightCornerX = dfRow['mouth_right_corner_x']
    mouthRightCornerY = dfRow['mouth_right_corner_y']

    mouthCenterTopLipX = dfRow['mouth_center_top_lip_x']
    mouthCenterTopLipY = dfRow['mouth_center_top_lip_y']
    mouthCenterBottomLipX = dfRow['mouth_center_bottom_lip_x']
    mouthCenterBottomLipY = dfRow['mouth_center_bottom_lip_y']

    noseTipX = dfRow['nose_tip_x']
    noseTipY = dfRow['nose_tip_y']

    try:
        print("Saving original image to png for item with index  " + str(i))

        binaryMaskRightEye = np.zeros((96, 96), dtype=np.int32)
        binaryMaskLeftEye = np.zeros((96, 96), dtype=np.int32)
        binaryMaskLips = np.zeros((96, 96), dtype=np.int32)

        fig = plt.imshow(item, aspect='equal', cmap='gray')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        # ax = plt.imshow(reshapedImages[i], cmap="gray", interpolation="nearest")


        originalImageDir = os.path.join('mydata/' + str(i) + '/images/')
        originalImageFilePath = str(i) + '.png'

        if not os.path.isdir(originalImageDir):
            os.makedirs(originalImageDir)

        plt.savefig(originalImageDir + originalImageFilePath, bbox_inches='tight', pad_inches=0)
        print("Successfully saved original image for item with index  " + str(i))
        # ax = fig.add_subplot(3, 3, i%9 + 1)
        # ax.imshow(reshapedImages[i], cmap="gray")
        # ax.scatter(noseTipX[i], noseTipY[i], c='red')
        # ax.scatter(rightEyeCenterX[i], rightEyeCenterY[i])

        # ax.plot([leftEyeInnerCornerX[i], leftEyeCenterX[i]], [leftEyeInnerCornerY[i], leftEyeCenterY[i]], 'ro-')
        # ax.plot([leftEyeCenterX[i], leftEyeOuterCornerX[i]], [leftEyeCenterY[i], leftEyeOuterCornerY[i]], 'ro-')

        # ax.plot([rightEyeInnerCornerX[i], rightEyeCenterX[i]], [rightEyeInnerCornerY[i], rightEyeCenterY[i]], 'ro-')
        # ax.plot([rightEyeCenterX[i], rightEyeOuterCornerX[i]], [rightEyeCenterY[i], rightEyeOuterCornerY[i]], 'ro-')
        #
        # ax.plot([leftEyebrowInnerEndX[i], leftEyebrowOuterEndX[i]], [leftEyebrowInnerEndY[i], leftEyebrowOuterEndY[i]], 'ro-')
        # ax.plot([rightEyebrowInnerEndX[i], rightEyebrowOuterEndX[i]], [rightEyebrowInnerEndY[i], rightEyebrowOuterEndY[i]], 'ro-')
        #
        # ax.plot([mouthLeftCornerX[i], mouthCenterTopLipX[i]], [mouthLeftCornerY[i], mouthCenterTopLipY[i]], 'ro-')
        # ax.plot([mouthCenterTopLipX[i], mouthRightCornerX[i]], [mouthCenterTopLipY[i], mouthRightCornerY[i]], 'ro-')
        # ax.plot([mouthRightCornerX[i], mouthCenterBottomLipX[i]], [mouthRightCornerY[i], mouthCenterBottomLipY[i]], 'ro-')
        # ax.plot([mouthCenterBottomLipX[i], mouthLeftCornerX[i]], [mouthCenterBottomLipY[i], mouthLeftCornerY[i]], 'ro-')

        # arc1 = ptc.Arc((0, 0), 16.0, 16.0,theta1 = -41.0,theta2 = 180.0+41.0)
        # ax.add_patch(arc1)

        print("Starting interpolation of keypoints for item with index " + str(i))

        xValuesTopLip = [mouthRightCornerX, mouthCenterTopLipX, mouthLeftCornerX]
        yValuesTopLip = [mouthRightCornerY, mouthCenterTopLipY, mouthLeftCornerY]
        xValuesBottomLip = [mouthRightCornerX, mouthCenterBottomLipX, mouthLeftCornerX]
        yValuesBottomLip = [mouthRightCornerY, mouthCenterBottomLipY, mouthLeftCornerY]
        csTopLip = CubicSpline(xValuesTopLip, yValuesTopLip)
        csBottomLip = CubicSpline(xValuesBottomLip, yValuesBottomLip)
        xLips = np.linspace(min(xValuesBottomLip), max(xValuesBottomLip), num=40, endpoint=True)
        # plt.plot(xLips, csBottomLip(xLips), color='red')
        # plt.plot(xLips, csTopLip(xLips), color='red')

        # xValuesLeftEyebrow = [leftEyebrowInnerEndX[i], (leftEyebrowInnerEndX[i] +  leftEyebrowOuterEndX[i]) / 2, leftEyebrowOuterEndX[i]]
        # yValuesLeftEyebrow = [leftEyebrowInnerEndY[i],  leftEyebrowInnerEndY[i] - 3, leftEyebrowOuterEndY[i]]
        # csLeftEyebrow = CubicSpline(xValuesLeftEyebrow, yValuesLeftEyebrow)
        # xLeftEyebrow = np.linspace(min(xValuesLeftEyebrow), max(xValuesLeftEyebrow), num=40, endpoint=True)
        # plt.plot(xLeftEyebrow, csLeftEyebrow(xLeftEyebrow), color='red')

        xValuesRightEye = [rightEyeOuterCornerX, rightEyeCenterX, rightEyeInnerCornerX]
        yValuesRightEyeTopArc = [rightEyeOuterCornerY, rightEyeCenterY - 3, rightEyeInnerCornerY]
        yValuesRightEyeBottomArc = [rightEyeOuterCornerY, rightEyeCenterY + 3, rightEyeInnerCornerY]
        csRightEyeTopArc = CubicSpline(xValuesRightEye, yValuesRightEyeTopArc)
        csRightEyeBottomArc = CubicSpline(xValuesRightEye, yValuesRightEyeBottomArc)
        xRightEye = np.linspace(min(xValuesRightEye), max(xValuesRightEye), num=40, endpoint=True)
        # plt.plot(xRightEye, csRightEyeTopArc(xRightEye), color='red')
        # plt.plot(xRightEye, csRightEyeBottomArc(xRightEye), color='red')

        xValuesLeftEye = [leftEyeInnerCornerX, leftEyeCenterX, leftEyeOuterCornerX]
        yValuesLeftEyeTopArc = [leftEyeInnerCornerY, leftEyeCenterY - 3, leftEyeOuterCornerY]
        yValuesLeftEyeBottomArc = [leftEyeInnerCornerY, leftEyeCenterY + 3, leftEyeOuterCornerY]
        csLeftEyeTopArc = CubicSpline(xValuesLeftEye, yValuesLeftEyeTopArc)
        csLeftEyeBottomArc = CubicSpline(xValuesLeftEye, yValuesLeftEyeBottomArc)
        xLeftEye = np.linspace(min(xValuesLeftEye), max(xValuesLeftEye), num=40, endpoint=True)
        # plt.plot(xLeftEye, csLeftEyeTopArc(xLeftEye), color='red')
        # plt.plot(xLeftEye, csLeftEyeBottomArc(xLeftEye), color='red')

        for rowIndex, row in enumerate(item):
            for columnIndex, greyscaleValues in enumerate(row):
                if columnIndex >= min(xRightEye) and columnIndex <= max(xRightEye) and rowIndex >= csRightEyeTopArc(
                        columnIndex) and rowIndex <= csRightEyeBottomArc(columnIndex):
                    binaryMaskRightEye[rowIndex][columnIndex] = 1
                else:
                    binaryMaskRightEye[rowIndex][columnIndex] = 0
                if columnIndex >= min(xLeftEye) and columnIndex <= max(xLeftEye) and rowIndex >= csLeftEyeTopArc(
                        columnIndex) and rowIndex <= csLeftEyeBottomArc(columnIndex):
                    binaryMaskLeftEye[rowIndex][columnIndex] = 1
                else:
                    binaryMaskLeftEye[rowIndex][columnIndex] = 0
                if columnIndex >= min(xLips) and columnIndex <= max(xLips) and rowIndex >= csTopLip(
                        columnIndex) and rowIndex <= csBottomLip(columnIndex):
                    binaryMaskLips[rowIndex][columnIndex] = 1
                else:
                    binaryMaskLips[rowIndex][columnIndex] = 0


        masksDir = os.path.join('mydata/' + str(i) + '/masks/')
        maskRightEyeFileName = str(i) + '.png'
        maskLeftEyeFileName = str(i) + str(i) + '.png'
        maskLipsEyeFileName = str(i) + str(i) + str(i) + '.png'

        if not os.path.isdir(masksDir):
            os.makedirs(masksDir)

        fig = plt.imshow(binaryMaskRightEye, aspect='equal', cmap='gray')
        plt.savefig(masksDir + maskRightEyeFileName, bbox_inches='tight', pad_inches=0)

        fig = plt.imshow(binaryMaskLeftEye, aspect='equal', cmap='gray')
        plt.savefig(masksDir + maskLeftEyeFileName, bbox_inches='tight', pad_inches=0)

        fig = plt.imshow(binaryMaskLips, aspect='equal', cmap='gray')
        plt.savefig(masksDir + maskLipsEyeFileName, bbox_inches='tight', pad_inches=0)

        # ax.set_title("input" + str(i))
        print("Successfully saved mask images for item with index " + str(i))
    except:
        print("problem with item with index - " + str(i))


def save_test_images(item, i):
    try:
        print("Saving test image to png for item with index  " + str(i))

        fig = plt.imshow(item, aspect='equal', cmap='gray')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        testImageDir = os.path.join('mytestdata/' + str(i) + '/images/')
        testImageFilePath = str(i) + '.png'

        if not os.path.isdir(testImageDir):
            os.makedirs(testImageDir)

        plt.savefig(testImageDir + testImageFilePath, bbox_inches='tight', pad_inches=0)
        print("Successfully saved test image for item with index  " + str(i))
    except:
        print("Problem saving test image with id - " + str(i))


if __name__ == "__main__":
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
    sess = tf.Session(config=config)

    TEST = False

    FTRAIN = "data/training.csv"
    FTEST  = "data/test.csv"
    FIdLookup = 'data/IdLookupTable.csv'

    X, df= load2d(TEST)

    reshapedImages = [np.reshape(n, (96, 96)) for n in X]
    executor = concurrent.futures.ProcessPoolExecutor(1)

    if TEST:
        futures = [executor.submit(save_test_images, item, index) for index, item in
                   enumerate(reshapedImages)]
        concurrent.futures.wait(futures)
    else:
        futures = [executor.submit(save_images_and_masks, item, index, df.iloc[index]) for index, item in enumerate(reshapedImages)]
        concurrent.futures.wait(futures)
    # for index, item in enumerate(reshapedImages):
    #     save_images_and_masks(item,index)
    # plt.show()
