import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def to_log_polar_coords(image):
    flags = cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS + cv.WARP_POLAR_LOG
    w, h = image.shape[::-1]
    centre = (w/2, h/2)
    R = (w-1)/2
    log_polar_img = cv.warpPolar(image, (w, h), centre, R, flags)
    return log_polar_img

def linear_transforms(image):
    w, h = image.shape[::-1]
    k = 1 + 0.05 * 10
    r = 3 * 10
    centre = (w/2, h/2)
    matrix = cv.getRotationMatrix2D(centre, r, k)
    transformed_image = cv.warpAffine(image, matrix, (w, h))
    cv.imwrite("rot_templ.jpg", transformed_image)
    return transformed_image

def correlation(img, template_img):
    result = cv.matchTemplate(img, template_img, cv.TM_CCOEFF_NORMED)
    plt.plot(), plt.imshow(result, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig('corr.png', bbox_inches='tight')

def main():
    img = cv.imread("main.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Can\'t read the image")
        return -1
       
    template = img.copy() 
    log_polar_img = to_log_polar_coords(img)
    cv.imwrite("main_polar.jpg", log_polar_img)
    template = linear_transforms(template)
    log_polar_template = to_log_polar_coords(template)
    w, h = log_polar_img.shape[::-1]
    new_img = np.zeros((2*h, 2*w), dtype='uint8')
    
    for k in (0, 1):
        for i, row in enumerate(log_polar_img):
            for j, pixel in enumerate(row):
                new_img[i+k*h][j] = pixel
                new_img[i+k*h][j + w] = pixel

    cv.imwrite("main_polar_collage.jpg", new_img)
    cv.imwrite("templ_polar.jpg", log_polar_template)
    correlation(new_img, log_polar_template)
    return 0


main()
