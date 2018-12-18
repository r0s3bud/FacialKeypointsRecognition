import numpy as np
import matplotlib.pyplot as plt
import imutils
import random
import os
from multiprocessing import Pool
import cv2
from shutil import copyfile
from scipy.ndimage import zoom

def augment_image(id):
    if (int(id) != 51):
        return

    dataPath = 'mydata/'
    augmentedDataPath = 'myaugmenteddata/'
    imageDir = str(id) + '/images/'
    imageFileName = str(id) + '.png'
    plt.axis('off')

    maskFolderPath = str(id) + '/masks/'
    maskFileNames = next(os.walk(dataPath + maskFolderPath))[2]

    img = plt.imread(dataPath + imageDir + imageFileName)

    # if int(id) % 4 == 0:
    #     angle = randomValue = random.randint(-20,20)
    #
    #     rotatedImg = imutils.rotate(img, angle)
    #
    #     for rowIndex, row in enumerate(rotatedImg):
    #         for columnIndex, column in enumerate(row):
    #             if column[3] == 0:
    #                 randomValue = random.uniform(0, 1)
    #                 rotatedImg[rowIndex][columnIndex]= [randomValue, randomValue, randomValue, randomValue]
    #
    #     if not os.path.isdir(augmentedDataPath + imageDir):
    #         os.makedirs(augmentedDataPath + imageDir)
    #
    #     fig = plt.imshow(rotatedImg)
    #     fig.axes.get_xaxis().set_visible(False)
    #     fig.axes.get_yaxis().set_visible(False)
    #     plt.savefig(augmentedDataPath + imageDir + 'test_' + imageFileName, bbox_inches='tight', pad_inches=0)
    #
    #     for maskFileName in maskFileNames:
    #         maskImg = plt.imread(dataPath + maskFolderPath + maskFileName)
    #         rotatedMaskImg = imutils.rotate(maskImg, angle)
    #
    #         for rowIndex, row in enumerate(rotatedMaskImg):
    #             for columnIndex, column in enumerate(row):
    #                 if column[3] < 0.99:
    #                     rotatedMaskImg[rowIndex][columnIndex] = [0, 0, 0, 1]
    #
    #         if not os.path.isdir(augmentedDataPath + maskFolderPath):
    #             os.makedirs(augmentedDataPath + maskFolderPath)
    #
    #         fig = plt.imshow(rotatedMaskImg)
    #         fig.axes.get_xaxis().set_visible(False)
    #         fig.axes.get_yaxis().set_visible(False)
    #         plt.savefig(augmentedDataPath + maskFolderPath + maskFileName, bbox_inches='tight', pad_inches=0)

    if (int(id) + 1) % 4 == 0:
        verticallyFlippedImg = cv2.flip(img, 1)

        if not os.path.isdir(augmentedDataPath + imageDir):
            os.makedirs(augmentedDataPath + imageDir)

        fig = plt.imshow(verticallyFlippedImg)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(augmentedDataPath + imageDir + imageFileName, bbox_inches='tight', pad_inches=0)

        for maskFileName in maskFileNames:
            maskImg = plt.imread(dataPath + maskFolderPath + maskFileName)
            flippedMaskImg = cv2.flip(maskImg, 1)

            if not os.path.isdir(augmentedDataPath + maskFolderPath):
                os.makedirs(augmentedDataPath + maskFolderPath)

            fig = plt.imshow(flippedMaskImg)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(augmentedDataPath + maskFolderPath + 'test_' + maskFileName, bbox_inches='tight', pad_inches=0)

    # if (int(id) + 2) % 4 == 0:
    #     imgChangedLuminosity = np.zeros((369,369,4), dtype=np.float32)
    #
    #     luminosityRatio = random.uniform(0.8,1.2)
    #
    #     for rowIndex, row in enumerate(img):
    #         for columnIndex, column in enumerate(row):
    #                 imgChangedLuminosity[rowIndex,columnIndex] = [column[0] * luminosityRatio,column[1] * luminosityRatio,column[2] * luminosityRatio,1]
    #
    #     if not os.path.isdir(augmentedDataPath + imageDir):
    #         os.makedirs(augmentedDataPath + imageDir)
    #
    #     fig = plt.imshow(imgChangedLuminosity)
    #     fig.axes.get_xaxis().set_visible(False)
    #     fig.axes.get_yaxis().set_visible(False)
    #     plt.savefig(augmentedDataPath + imageDir + imageFileName, bbox_inches='tight', pad_inches=0)
        #
        # for maskFileName in maskFileNames:
        #     if not os.path.isdir(augmentedDataPath + maskFolderPath):
        #         os.makedirs(augmentedDataPath + maskFolderPath)
        #
        #     copyfile(dataPath + maskFolderPath + maskFileName, augmentedDataPath + maskFolderPath + maskFileName)

    # if (int(id) + 3) % 4 == 0:
    #     zoom_factor = random.uniform(0.9,1.1)
    #     h, w = img.shape[:2]
    # 
    #     zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    # 
    #     if zoom_factor < 1:
    # 
    #         zh = int(np.round(h * zoom_factor))
    #         zw = int(np.round(w * zoom_factor))
    #         top = (h - zh) // 2
    #         left = (w - zw) // 2
    # 
    #         out = np.zeros_like(img)
    #         out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple)
    # 
    #         for maskFileName in maskFileNames:
    #             maskImg = plt.imread(dataPath + maskFolderPath + maskFileName)
    #             outMask = np.zeros_like(maskImg)
    #             outMask[top:top + zh, left:left + zw] = zoom(maskImg, zoom_tuple)
    # 
    #             for rowIndex, row in enumerate(outMask):
    #                 for columnIndex, column in enumerate(row):
    #                     if column[3] < 0.99:
    #                         outMask[rowIndex][columnIndex] = [0, 0, 0, 1]
    # 
    #             if not os.path.isdir(augmentedDataPath + maskFolderPath):
    #                 os.makedirs(augmentedDataPath + maskFolderPath)
    # 
    #             fig = plt.imshow(outMask)
    #             fig.axes.get_xaxis().set_visible(False)
    #             fig.axes.get_yaxis().set_visible(False)
    #             plt.savefig(augmentedDataPath + maskFolderPath + maskFileName, bbox_inches='tight', pad_inches=0)
    # 
    #     elif zoom_factor > 1:
    # 
    #         zh = int(np.round(h / zoom_factor))
    #         zw = int(np.round(w / zoom_factor))
    #         top = (h - zh) // 2
    #         left = (w - zw) // 2
    # 
    #         out = zoom(img[top:top + zh, left:left + zw], zoom_tuple)
    # 
    #         trim_top = ((out.shape[0] - h) // 2)
    #         trim_left = ((out.shape[1] - w) // 2)
    #         out = out[trim_top:trim_top + h, trim_left:trim_left + w]
    # 
    #         for maskFileName in maskFileNames:
    #             maskImg = plt.imread(dataPath + maskFolderPath + maskFileName)
    #             outMask = np.zeros_like(maskImg)
    # 
    #             outMask = zoom(maskImg[top:top + zh, left:left + zw], zoom_tuple)
    #             outMask = outMask[trim_top:trim_top + h, trim_left:trim_left + w]
    # 
    #             for rowIndex, row in enumerate(outMask):
    #                 for columnIndex, column in enumerate(row):
    #                     if column[3] < 0.99:
    #                         outMask[rowIndex][columnIndex] = [0, 0, 0, 1]
    # 
    #             if not os.path.isdir(augmentedDataPath + maskFolderPath):
    #                 os.makedirs(augmentedDataPath + maskFolderPath)
    # 
    #             fig = plt.imshow(outMask)
    #             fig.axes.get_xaxis().set_visible(False)
    #             fig.axes.get_yaxis().set_visible(False)
    #             plt.savefig(augmentedDataPath + maskFolderPath + maskFileName, bbox_inches='tight', pad_inches=0)
    # 
    #     for rowIndex, row in enumerate(out):
    #         for columnIndex, column in enumerate(row):
    #             if column[3] == 0:
    #                 randomValue = random.uniform(0, 1)
    #                 out[rowIndex][columnIndex] = [randomValue, randomValue, randomValue, randomValue]
    # 
    #     if not os.path.isdir(augmentedDataPath + imageDir):
    #         os.makedirs(augmentedDataPath + imageDir)
    # 
    #     fig = plt.imshow(out)
    #     fig.axes.get_xaxis().set_visible(False)
    #     fig.axes.get_yaxis().set_visible(False)
    #     plt.savefig(augmentedDataPath + imageDir + imageFileName, bbox_inches='tight', pad_inches=0)
  # if (int(id) + 3) % 4 == 0:
    #     zoom_factor = random.uniform(0.9,1.1)
    #     h, w = img.shape[:2]
    #
    #     zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    #
    #     if zoom_factor < 1:
    #
    #         zh = int(np.round(h * zoom_factor))
    #         zw = int(np.round(w * zoom_factor))
    #         top = (h - zh) // 2
    #         left = (w - zw) // 2
    #
    #         out = np.zeros_like(img)
    #         out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple)
    #
    #         for maskFileName in maskFileNames:
    #             maskImg = plt.imread(dataPath + maskFolderPath + maskFileName)
    #             outMask = np.zeros_like(maskImg)
    #             outMask[top:top + zh, left:left + zw] = zoom(maskImg, zoom_tuple)
    #
    #             for rowIndex, row in enumerate(outMask):
    #                 for columnIndex, column in enumerate(row):
    #                     if column[3] < 0.99:
    #                         outMask[rowIndex][columnIndex] = [0, 0, 0, 1]
    #
    #             if not os.path.isdir(augmentedDataPath + maskFolderPath):
    #                 os.makedirs(augmentedDataPath + maskFolderPath)
    #
    #             fig = plt.imshow(outMask)
    #             fig.axes.get_xaxis().set_visible(False)
    #             fig.axes.get_yaxis().set_visible(False)
    #             plt.savefig(augmentedDataPath + maskFolderPath + maskFileName, bbox_inches='tight', pad_inches=0)
    #
    #     elif zoom_factor > 1:
    #
    #         zh = int(np.round(h / zoom_factor))
    #         zw = int(np.round(w / zoom_factor))
    #         top = (h - zh) // 2
    #         left = (w - zw) // 2
    #
    #         out = zoom(img[top:top + zh, left:left + zw], zoom_tuple)
    #
    #         trim_top = ((out.shape[0] - h) // 2)
    #         trim_left = ((out.shape[1] - w) // 2)
    #         out = out[trim_top:trim_top + h, trim_left:trim_left + w]
    #
    #         for maskFileName in maskFileNames:
    #             maskImg = plt.imread(dataPath + maskFolderPath + maskFileName)
    #             outMask = np.zeros_like(maskImg)
    #
    #             outMask = zoom(maskImg[top:top + zh, left:left + zw], zoom_tuple)
    #             outMask = outMask[trim_top:trim_top + h, trim_left:trim_left + w]
    #
    #             for rowIndex, row in enumerate(outMask):
    #                 for columnIndex, column in enumerate(row):
    #                     if column[3] < 0.99:
    #                         outMask[rowIndex][columnIndex] = [0, 0, 0, 1]
    #
    #             if not os.path.isdir(augmentedDataPath + maskFolderPath):
    #                 os.makedirs(augmentedDataPath + maskFolderPath)
    #
    #             fig = plt.imshow(outMask)
    #             fig.axes.get_xaxis().set_visible(False)
    #             fig.axes.get_yaxis().set_visible(False)
    #             plt.savefig(augmentedDataPath + maskFolderPath + maskFileName, bbox_inches='tight', pad_inches=0)
    #
    #     for rowIndex, row in enumerate(out):
    #         for columnIndex, column in enumerate(row):
    #             if column[3] == 0:
    #                 randomValue = random.uniform(0, 1)
    #                 out[rowIndex][columnIndex] = [randomValue, randomValue, randomValue, randomValue]
    #
    #     if not os.path.isdir(augmentedDataPath + imageDir):
    #         os.makedirs(augmentedDataPath + imageDir)
    #
    #     fig = plt.imshow(out)
    #     fig.axes.get_xaxis().set_visible(False)
    #     fig.axes.get_yaxis().set_visible(False)
    #     plt.savefig(augmentedDataPath + imageDir + imageFileName, bbox_inches='tight', pad_inches=0)
if __name__ == '__main__':
    dataPath = 'mydata/'

    imageAndMaskIds = next(os.walk(dataPath))[1]

    pool = Pool(os.cpu_count())
    pool.map(augment_image, imageAndMaskIds)

