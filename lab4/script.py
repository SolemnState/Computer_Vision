import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

RESULT = 1

def dft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def to_log_polar_coords(image):
    flags = cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS + cv.WARP_POLAR_LOG
    w, h = image.shape[::-1]
    centre = (w/2, h/2)
    max_radius = np.sqrt((w/2)**2+(h/2)**2)
    log_polar_img = cv.warpPolar(image, (w, h), centre, max_radius, flags)
    #log_polar_img = cv.linearPolar(image, centre, max_radius, flags)
    return log_polar_img

def zoom_image(image, percents):
    w, h = image.shape[::-1]
    centre = (w/2, h/2)
    matrix = cv.getRotationMatrix2D(centre, 0, (percents/100))
    transformed_image = cv.warpAffine(image, matrix, (w,h))
    return transformed_image

def find_correlation(image, template_image):
    global RESULT
    m_i = dft(image)
    if RESULT == 1:
        cv.imwrite("DFT_MAIN.jpg", m_i)
    w, h = image.shape[::-1]
    t_i = dft(template_image)
    cv.imwrite("DFT_TEMPLATE" + str(RESULT) +".jpg", t_i)
    log_polar_main = to_log_polar_coords(m_i)
    if RESULT == 1:
        cv.imwrite("log_polar_main_img.jpg", log_polar_main)
    log_polar_templ = to_log_polar_coords(t_i)
    cv.imwrite("log_polar_template_img" + str(RESULT) + ".jpg", log_polar_templ)
    m_i = dft(log_polar_main)
    if RESULT == 1:
        cv.imwrite("main_log_polar_dft.jpg", m_i)
    t_i = dft(log_polar_templ)
    cv.imwrite("template_log_polar_dft" + str(RESULT) + ".jpg", t_i)
    avg_B = np.mean(m_i)
    avg_B_m = np.mean(t_i)
    sum_num = 0
    sum_den_B = 0
    sum_den_B_m = 0
    for i in range(h):
        for j in range(w):
            sum_num += (m_i[i][j] - avg_B) * (t_i[i][j] - avg_B_m)
            sum_den_B += (m_i[i][j] - avg_B) ** 2
            sum_den_B_m += (t_i[i][j] - avg_B_m) ** 2
    corr_coeff = sum_num / np.sqrt(sum_den_B * sum_den_B_m)
    print("\t Correlation coefficient for test number " + str(RESULT) +": ", str('{:.3f}'.format(corr_coeff)))
    RESULT += 1

if __name__ == "__main__":
    main_templ = cv.imread("main_template.jpg", cv.IMREAD_GRAYSCALE)
    main_zoom_20 = zoom_image(main_templ, 120)
    cv.imwrite("main_zoom_20.jpg", main_zoom_20)
    main_zoom_50 = zoom_image(main_templ, 150)
    cv.imwrite("main_zoom_50.jpg", main_zoom_50)
    main_zoom_100 = zoom_image(main_templ, 200)
    cv.imwrite("main_zoom_100.jpg", main_zoom_100)
    templ20 = cv.imread("main_template_20.jpg", cv.IMREAD_GRAYSCALE)
    templ40 = cv.imread("main_template_40.jpg", cv.IMREAD_GRAYSCALE)
    templ60 = cv.imread("main_template_60.jpg", cv.IMREAD_GRAYSCALE)
    foreign = cv.imread("foreign.jpg", cv.IMREAD_GRAYSCALE)
    list_of_imgs = [main_templ, main_zoom_20, main_zoom_50, main_zoom_100, templ20, templ40, templ60, foreign]
    list(map(lambda img: find_correlation(main_templ, img), list_of_imgs))
    