import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import pi, sqrt, atan, sin, cos

def normalization(image):
    ret, thresh = cv.threshold(image, 127, 255, 0)
    M = cv.moments(thresh)
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    print("Центр массы изображения по осям X и Y:" +'\n'+ str(int(cX)) + "  " + str(int(cY)))
    sum_x = 0
    sum_y = 0
    denominator = 0
    B = 0
    C = 0
    D = 0
    K = 10
    w, h = image.shape[::-1]
    centre = (w/2, h/2)
    new_image = np.zeros((h, w), dtype='uint8')
    
    for y, row in enumerate(image):
            for x, pixel in enumerate(row):
                if pixel != 0:
                    B += pixel * ((x - cX)**2 - (y - cY)**2)
                    C += pixel * 2 * (x - cX) * (y - cY)
                    D += pixel * ((x - cX)**2 + (y - cY)**2)
    mu = sqrt((D + sqrt(C**2 + B**2)) / (D - sqrt(C**2 + B**2)))
    omega = 0.5 * atan(C/B) + pi
    print("Направление сжатия изображения:" + '\n' + str('{:.3f}'.format(omega)))
    print("Величина сжатия изображения:" + '\n' + str('{:.3f}'.format(mu)))
    numerator_M = 0
    denominator_M = 0 
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            if pixel != 0:
                coord_x = (1/mu) * ((x - cX) * cos(-omega) - (y - cY) * sin(-omega)) * cos(omega)- \
                     ((x - cX) * sin(-omega) + (y - cY) * cos(-omega))* sin(omega)
                coord_y = (1/mu) * ((x - cX) * cos(-omega) - (y - cY) * sin(-omega)) * sin(omega)+ \
                     ((x - cX) * sin(-omega) + (y - cY) * cos(-omega))* cos(omega)
                new_image[int(coord_y) + int(cY)][int(coord_x) + int(cX)] = pixel
                numerator_M += pixel * sqrt(coord_x**2 + coord_y**2)
                denominator_M +=  pixel
       
    M = numerator_M / (K * denominator_M)
    print("Коэффициент равномерного масштабирования:" + '\n' + str('{:.3f}'.format(M)))
    matrix = cv.getRotationMatrix2D(centre, 0, 1/M)
    new_image = cv.warpAffine(new_image, matrix, (w, h))
    return new_image

if __name__ == "__main__":
    img1 = cv.imread("image_1_18.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("image_2_18.jpg", cv.IMREAD_GRAYSCALE)
    img3 = cv.imread("image_3_18.jpg", cv.IMREAD_GRAYSCALE)
    
    if ((img1 is None) or (img2 is None) or (img3 is None)):
        print("Can\'t read the image")
        quit()

    imgs = list(map(normalization, [img1, img2, img3]))
    cv.imwrite("transformed1.jpg", imgs[0])
    cv.imwrite("transformed2.jpg", imgs[1])
    cv.imwrite("transformed3.jpg", imgs[2])

