# from google.cloud import vision
# import pandas as pd
import cv2
import numpy as np
import math

# from gvision import GVisionAPI

# authfile = '/home/lutsai/.config/gcloud/application_default_credentials.json'
# gvision = GVisionAPI(authfile)

# [END vision_python_migration_import]

path_yellow = '/lnet/work/people/lutsai/pythonProject/pages/CTX193102237_page_1.png'
path_dirty = '/lnet/work/people/lutsai/pythonProject/pages/CTX194604301_page_0.png'
path_draw = '/lnet/work/people/lutsai/pythonProject/pages/CTX198702238A_page_24.png'
path_bleak = '/lnet/work/people/lutsai/pythonProject/pages/CTX199706756_page_67.png'
path_skew = '/lnet/work/people/lutsai/pythonProject/pages/CTX198402735_page_0.png'


image = cv2.imread(path_skew)
# cv2.imshow("img", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("img_gray", image_gray)

ret, thresh_image = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow("img_tresh", thresh_image)

thresh_gauss_big = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 49, 15)
thresh_gauss_small = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 9, 5)
# cv2.imshow("img_tresh_gauss", thresh_gauss)

ret2, th2 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("img_tresh_otsu", th2)

blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("img_tresh_gauss_otsu", th3)

gray_concat = np.concatenate((image_gray, thresh_image), axis=1)
cv2.imshow("img_gray_stack", gray_concat)

tresh_concat = np.concatenate((thresh_gauss_small, thresh_gauss_big), axis=1)
cv2.imshow("img_tresh_stack", tresh_concat)

otsu_concat = np.concatenate((th2, th3), axis=1)
cv2.imshow("img_otsu_stack", otsu_concat)


thresh_closed = cv2.morphologyEx(thresh_gauss_big, cv2.MORPH_CLOSE, np.ones((1, 10)))

# Find contours
contours = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).

angles = []  # List of line angles.

# Iterate the contours and fit a line for each contour
# Remark: consider ignoring small contours
for c in contours:
    vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01) # Fit line
    w = image_gray.shape[1]
    cv2.line(image_gray, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 255, 0))  # Draw the line for testing
    ang = (180/np.pi)*math.atan2(vy, vx) # Compute the angle of the line.
    angles.append(ang)

angles = np.array(angles)  # Convert angles to NumPy array.

# Remove outliers and
lo_val, up_val = np.percentile(angles, (40, 60))  # Get the value of lower and upper 40% of all angles (mean of only 10 angles)
mean_ang = np.mean(angles[np.where((angles >= lo_val) & (angles <= up_val))])

print(f'mean_ang = {mean_ang}')  # -0.2424

M = cv2.getRotationMatrix2D((image_gray.shape[1]//2, image_gray.shape[0]//2), mean_ang, 1)  # Get transformation matrix - for rotating by mean_ang

img_deskew = cv2.warpAffine(image_gray, M, (image_gray.shape[1], image_gray.shape[0]), cv2.INTER_CUBIC) # Rotate the image
gray_concat = np.concatenate((image_gray, img_deskew), axis=1)
cv2.imshow("img_skew", gray_concat)


cv2.waitKey(0)
cv2.destroyAllWindows()