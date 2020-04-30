import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

CORR_RESULT = 1

def template_matching(image,template_image):
    global CORR_RESULT
    temp_image = image.copy()
    result = cv.matchTemplate(temp_image, template_image, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    w, h = template_image.shape[::-1]
    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    cv.rectangle(temp_image, max_loc, bottom_right, 255, 2)
    plt.plot(), plt.imshow(result, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig('corr' + str(CORR_RESULT) + '.png', bbox_inches='tight')
    CORR_RESULT += 1

def corr_rotated(image, template_image):
    rotation_angle = np.arange(-10, 10, 2)
    w1, h1 = image.shape[::-1]
    w2, h2 = template_image.shape[::-1]
    centre1 = (w1/2, h1/2)
    centre2 = (w2/2, h2/2)
    list_of_main_mx = map(lambda angle: cv.getRotationMatrix2D(centre1, angle, 1.0),
                          rotation_angle)
    list_of_template_mx = map(lambda angle: cv.getRotationMatrix2D(centre2, angle, 1.0),
                              rotation_angle)
    rotated_main_images = map(lambda l: cv.warpAffine(image, l, (w1, h1)),
                              list_of_main_mx)
    rotated_template_images = map(lambda l: cv.warpAffine(template_image, l, (w2, h2)),
                                  list_of_template_mx)
    for img, template_img in zip(rotated_main_images, rotated_template_images):
        template_matching(img, template_img)

def corr_scaled(image, template_image):
    scaling = np.arange(0.9, 1.1, 0.025)
    w1, h1 = image.shape[::-1]
    w2, h2 = template_image.shape[::-1]
    shape_of_main = [(int(w1*i), int(h1*i)) for i in scaling]
    shape_of_template = [(int(w2*i), int(h2*i)) for i in scaling]
    list_of_main_images = map(lambda scale_arg: cv.resize(image, scale_arg),
                              shape_of_main)
    list_of_template_images = map(lambda scale_arg: cv.resize(template_image, scale_arg),
                                  shape_of_template)
    for img, template_img in zip(list_of_main_images, list_of_template_images):
        template_matching(img, template_img)

def main():
    img = cv.imread("main.jpg", cv.IMREAD_GRAYSCALE)
    template1 = cv.imread("own_21.jpeg", cv.IMREAD_GRAYSCALE)
    template2 = cv.imread("foreign_11.jpeg", cv.IMREAD_GRAYSCALE)
    if ((img is None) or (template1 is None) or (template2 is None)):
        print("Can\'t read one of the images")
        return -1
    template_matching(img, template1)
    template_matching(img, template2)
    corr_rotated(img, template1)
    corr_scaled(img, template1)
    return 0

main()
