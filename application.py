import math
import os
import numpy as np
from matplotlib import pyplot as plt
from cv2 import mean
import cv2 as cv


image = cv.imread('soobin.jpg')
height, width = image.shape[:2]


def showResult(nrow=0, ncol=0, res_stack=None):
    plt.figure(figsize=(12, 12))
    for idx, (lbl, img) in enumerate(res_stack):
        plt.subplot(nrow, ncol, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
    plt.show()


gray_ocv = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray_avg = np.dot(image, [0.33, 0.33, 0.33])


def grayScale():

    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    max_cha = max(np.amax(b), np.amax(g), np.amax(r))
    min_cha = min(np.amin(b), np.amin(g), np.amin(r))
    gray_lig = np.dot(image, [(max_cha+min_cha)/2,
                              (max_cha+min_cha)/2, (max_cha+min_cha)/2])

    gray_lum = np.dot(image, [0.07, 0.71, 0.21])
    gray_wag = np.dot(image, [0.114, 0.587, 0.299])

    gray_labels = ['gray open cv', 'gray average', 'gray lightness',
                   'gray luminosity', 'gray weighted average']
    gray_img = [gray_ocv, gray_avg, gray_lig, gray_lum, gray_wag]
    showResult(3, 2, zip(gray_labels, gray_img))


def threshHold():
    thresh = 100
    thresh_image = gray_ocv.copy()
    for i in range(height):
        for j in range(width):
            if thresh_image[i, j] > thresh:
                thresh_image[i, j] = 255
            else:
                thresh_image[i, j] = 0

    _, bin_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_BINARY)
    _, inv_bin_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_BINARY_INV)
    _, trunc_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_TRUNC)
    _, tozero_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_TOZERO)
    _, inv_tozero_thresh = cv.threshold(
        gray_ocv, 100, 255, cv.THRESH_TOZERO_INV)
    _, otsu_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_OTSU)

    thresh_labels = ['bin', 'inv_bin', 'trunc', 'tozero', 'tozero_inv', 'otsu']
    thresh_img = [bin_thresh, inv_bin_thresh, trunc_thresh,
                  tozero_thresh, inv_tozero_thresh, otsu_thresh]
    showResult(3, 3, zip(thresh_labels, thresh_img))

# image processing filter


def useFilter():
    def manual_mean_filter(source, ksize):
        np_source = np.array(source)
        for i in range(height - ksize - 1):
            for j in range(width - ksize - 1):
                matrix = np.array(
                    np_source[i: (i + ksize), j: (j + ksize)]).flatten()
                mean = np.mean(matrix)

                np_source[i + ksize//2, j + ksize//2] = mean
        return np_source

    def manual_median_filter(source, ksize):
        np_source = np.array(source)
        for i in range(height - ksize - 1):
            for j in range(width-ksize - 1):
                matrix = np.array(
                    np_source[i:(i+ksize), j:(j+ksize)]).flatten()
                median = np.median(matrix)
                np_source[i + ksize//2, j + ksize//2] = median
        return np_source

    b, g, r = cv.split(image)
    ksize = 5

    b_mean = manual_mean_filter(b, ksize)
    g_mean = manual_mean_filter(g, ksize)
    r_mean = manual_mean_filter(r, ksize)

    b_median = manual_median_filter(b, ksize)
    g_median = manual_median_filter(g, ksize)
    r_median = manual_median_filter(r, ksize)

    mmean_filter = cv.merge((r_mean, b_mean, g_mean))
    mmedian_filter = cv.merge((r_median, b_median, g_median))

    filter_image = gray_ocv.copy()

    mean_blur = cv.blur(filter_image, (5, 5))
    median_blur = cv.medianBlur(filter_image, 5)
    gaussian_blur = cv.GaussianBlur(filter_image, (5, 5), 2.0)
    bilateral_blur = cv.bilateralFilter(filter_image, 5, 150, 150)

    filter_labels = ['manual mean', 'manual median', 'mean filter',
                     'median filter', 'gauss filter', 'bilateral filter']
    filter_images = [mmean_filter, mmedian_filter, mean_blur,
                     median_blur, gaussian_blur, bilateral_blur]
    showResult(3, 2, zip(filter_labels, filter_images))


# ================================================================================================================================================================================================================================
# EDGE DETECTION
image2 = cv.imread('bear.jpg')
igray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
height, width = image2.shape[:2]


def laplace():
    laplace_uintu08 = cv.Laplacian(igray, cv.CV_8U)
    laplace_uintu16 = cv.Laplacian(igray, cv.CV_16S)
    laplace_uintu32 = cv.Laplacian(igray, cv.CV_32F)
    laplace_uintu64 = cv.Laplacian(igray, cv.CV_64F)

    laplace_labels = ['laplace 8-bit', 'laplace 16-bit',
                      'laplace 32-bit', 'laplace 64-bit']
    laplace_images = [laplace_uintu08, laplace_uintu16,
                      laplace_uintu32, laplace_uintu64]
    showResult(2, 2, zip(laplace_labels, laplace_images))


def sobel():
    def calculateSobel(source, kernel, ksize):
        res_matrix = np.array(source)
        for i in range(height-ksize-1):
            for j in range(width-ksize-1):
                patch = source[i:(i+ksize), j:(j+ksize)].flatten()
                result = np.convolve(patch, kernel, 'valid')
                res_matrix[i+ksize//2, j+ksize//2] = result[0]
        return res_matrix

    kernel_x = np.array([
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    ])

    kernel_y = np.array([
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    ])

    ksize = 3
    manual_sobel_x = igray.copy()
    manual_sobel_y = igray.copy()

    manual_sobel_x = calculateSobel(manual_sobel_x, kernel_x, ksize)
    manual_sobel_y = calculateSobel(manual_sobel_y, kernel_y, ksize)

    sobel_x = cv.Sobel(igray, cv.CV_32F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(igray, cv.CV_32F, 0, 1, ksize=3)

    sobel_labels = ['msobelx', 'msobely', 'sobelx', 'sobely']
    sobel_imgs = [manual_sobel_x, manual_sobel_y, sobel_x, sobel_y]
    showResult(2, 2, zip(sobel_labels, sobel_imgs))

    merged_sobel = np.sqrt(np.square(sobel_x)+np.square(sobel_y))

    merged_sobel *= 225.0/merged_sobel.max()

    manual_merged_sobel = np.bitwise_or(manual_sobel_x, manual_sobel_y)

    manual_merged_sobel = np.uint(np.absolute(manual_merged_sobel))

    merged_sobel_labels = ['manual merged', 'merged']
    merged_sobel_imgs = [manual_merged_sobel, merged_sobel]
    showResult(2, 2, zip(merged_sobel_labels, merged_sobel_imgs))


def canny():
    canny_050100 = cv.Canny(igray, 50, 100)
    canny_050150 = cv.Canny(igray, 50, 150)
    canny_075100 = cv.Canny(igray, 75, 100)
    canny_075225 = cv.Canny(igray, 75, 225)

    canny_labels = ['50 100', '50 150', '75 100', '75 225']
    canny_imgs = [canny_050100, canny_050150, canny_075100, canny_075225]
    showResult(2, 2, zip(canny_labels, canny_imgs))


# ================================================================================================================================================================================================================================
# SHAPE DETECTION
def shapeDetect():
    font = cv.FONT_HERSHEY_COMPLEX
    img = cv.imread("shapes.png", cv.IMREAD_GRAYSCALE)
    _, threshold = cv.threshold(img, 240, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
        cv.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 3:
            cv.putText(img, "Triangle", (x, y), font, 1, (0))
        elif len(approx) == 4:
            cv.putText(img, "Rectangle", (x, y), font, 1, (0))
        elif len(approx) == 5:
            cv.putText(img, "Pentagon", (x, y), font, 1, (0))
        elif 6 < len(approx) < 15:
            cv.putText(img, "Ellipse", (x, y), font, 1, (0))
        else:
            cv.putText(img, "Circle", (x, y), font, 1, (0))

    cv.imshow("shapes", img)
    cv.imshow("Threshold", threshold)
    cv.waitKey(0)
    cv.destroyAllWindows()


# ================================================================================================================================================================================================================================
# PATERN RECOGNITION
def paternRecog():
    image_obj = cv.imread('logo.png')
    image_scn = cv.imread('logo2.jpg')

    SIFT = cv.xfeatures2d.SIFT_create()
    SURF = cv.xfeatures2d.SURF_create()
    ORB = cv.ORB_create()
    AKAZE = cv.AKAZE_create()
    sift_kp_obj, sift_ds_obj = SIFT.detectAndCompute(image_obj, None)
    sift_kp_scn, sift_ds_scn = SIFT.detectAndCompute(image_scn, None)

    surf_kp_obj, surf_ds_obj = SURF.detectAndCompute(image_obj, None)
    surf_kp_scn, surf_ds_scn = SURF.detectAndCompute(image_scn, None)

    orb_kp_obj, orb_ds_obj = ORB.detectAndCompute(image_obj, None)
    orb_kp_scn, orb_ds_scn = ORB.detectAndCompute(image_scn, None)

    akaze_kp_obj, akaze_ds_obj = AKAZE.detectAndCompute(image_obj, None)
    akaze_kp_scn, akaze_ds_scn = AKAZE.detectAndCompute(image_scn, None)

    sift_ds_obj = np.float32(sift_ds_obj)
    sift_ds_scn = np.float32(sift_ds_scn)

    surf_ds_obj = np.float32(surf_ds_obj)
    surf_ds_scn = np.float32(surf_ds_scn)

    akaze_ds_obj = np.float32(akaze_ds_obj)
    akaze_ds_scn = np.float32(akaze_ds_scn)

    flann = cv.FlannBasedMatcher(dict(algorithm=1), dict(checks=50))
    bfmatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    sift_match = flann.knnMatch(sift_ds_obj, sift_ds_scn, 2)
    surf_match = flann.knnMatch(surf_ds_obj, surf_ds_scn, 2)
    akaze_match = flann.knnMatch(akaze_ds_obj, akaze_ds_scn, 2)

    orb_match = bfmatcher.match(orb_ds_obj, orb_ds_scn)

    orb_match = sorted(orb_match, key=lambda x: x.distance)

    def createMask(mask, match):
        for i, (fm, sm) in enumerate(match):
            if fm.distance < 0.7*sm.distance:
                mask[i] = [1, 0]

        return mask

    sift_matchesmask = [[0, 0] for i in range(0, len(sift_match))]
    surf_matchesmask = [[0, 0] for i in range(0, len(surf_match))]
    akaze_matchesmask = [[0, 0] for i in range(0, len(akaze_match))]

    sift_matchesmask = createMask(sift_matchesmask, sift_match)
    surf_matchesmask = createMask(surf_matchesmask, surf_match)
    akaze_matchesmask = createMask(akaze_matchesmask, akaze_match)

    sift_res = cv.drawMatchesKnn(
        image_obj, sift_kp_obj,
        image_scn, sift_kp_scn,
        sift_match, None,
        matchColor=[255, 0, 0], singlePointColor=[0, 255, 0],
        matchesMask=sift_matchesmask
    )

    surf_res = cv.drawMatchesKnn(
        image_obj, surf_kp_obj,
        image_scn, surf_kp_scn,
        surf_match, None,
        matchColor=[255, 0, 0], singlePointColor=[0, 255, 0],
        matchesMask=surf_matchesmask
    )

    akaze_res = cv.drawMatchesKnn(
        image_obj, akaze_kp_obj,
        image_scn, akaze_kp_scn,
        akaze_match, None,
        matchColor=[255, 0, 0], singlePointColor=[0, 255, 0],
        matchesMask=akaze_matchesmask
    )

    orb_res = cv.drawMatches(
        image_obj, orb_kp_obj,
        image_scn, orb_kp_scn,
        orb_match[:20], None,
        matchColor=[255, 0, 0], singlePointColor=[0, 255, 0],
        flags=2
    )

    matching_labels = ['sift', 'surf', 'akaze', 'orb']
    matching_imgs = [sift_res, surf_res, akaze_res, orb_res]

    plt.figure(figsize=(12, 12))

    for i, (lbl, img) in enumerate(zip(matching_labels, matching_imgs)):
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)

    plt.show()


# ================================================================================================================================================================================================================================
# FACE RECOGNITION
classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# TRAIN


def faceRecog():
    train_path = 'images/train'
    tdir = os.listdir(train_path)

    face_list = []
    class_list = []

    for index, train_dir in enumerate(tdir):
        for image_path in os.listdir(f'{train_path}/{train_dir}'):
            path = f'{train_path}/{train_dir}/{image_path}'
            if path.split('.')[1] != 'db':
                gray = cv.imread(path, 0)
                faces = classifier.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5)
                if len(faces) < 1:
                    continue
                for face_rect in faces:
                    x, y, w, h = face_rect
                    face_image = gray[y:y+w, x:x+h]
                    face_list.append(face_image)
                    class_list.append(index)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))

    # TEST

    test_path = 'images/test'
    for path in os.listdir(test_path):
        full_path = f'{test_path}/{path}'
        image = cv.imread(full_path)
        igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(
            igray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) < 1:
            continue

        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = gray[y:y+w, x:x+h]
            res, conf = face_recognizer.predict(face_image)
            conf = math.floor(conf*100)/100
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            text = f'{tdir[res]} {str(conf)}%'
            cv.putText(image, text, (x, y-10),
                       cv.FONT_HERSHEY_PLAIN, 1.5, (0.255, 0), 1)
            cv.imshow('result', image)
            cv.waitKey(0)
    cv.destroyAllWindows()

# ================================================================================================================================================================================================================================
# EDGE DETECTION MENU


def edgeDetect():
    choice = -1
    while choice != 0:
        print("1. laplace")
        print("2. sobel")
        print("3. Canny")
        print("4. back")
        choice = int(input("Choose> "))
        if choice == 1:
            laplace()
        elif choice == 2:
            sobel()
        elif choice == 3:
            canny()
        elif choice == 4:
            return
        else:
            print("invalid option [1-4]")


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMAGE PROCESING MENU


def imageProcessing():
    choice = -1
    while choice != 0:
        print("1. Gray Scale")
        print("2. treshHold")
        print("3. Filter")
        print("4 back")
        choice = int(input("Choose> "))
        if choice == 1:
            grayScale()
        elif choice == 2:
            threshHold()
        elif choice == 3:
            useFilter()
        elif choice == 4:
            return
        else:
            print("invalid option [1-4]")


# ============================================================================================================================================
# Main Menu


def menu():
    choice = -1
    while(choice != 0):
        print("1. Image Procesing")
        print("2. Edge Detection")
        print("3. Shape Detection")
        print("4. Patern Recognition")
        print("5. Face Detection")
        print("0. Exit ")
        choice = int(input("choose> "))
        if choice == 1:
            imageProcessing()
        elif choice == 2:
            edgeDetect()
        elif choice == 3:
            shapeDetect()
        elif choice == 4:
            paternRecog()
        elif choice == 5:
            faceRecog()
        elif choice == 0:
            print("Thanks see yaa again!")
            break
        else:
            print("invalid option [1-5]")


# MAIN MENU CALLED
menu()
