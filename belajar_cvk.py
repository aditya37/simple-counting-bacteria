import numpy as np
import cv2

image_origin = cv2.imread("321.png")
image_contorus = image_origin.copy()

# store counted colonies
counter = {}

image_to_process = image_origin.copy()

# convert to binary
ret,thresh = cv2.threshold(image_to_process,127,255,cv2.THRESH_BINARY)
cv2.imwrite("image_biner.png",thresh)

# image_erosi
erosion = cv2.erode(thresh,(5,5),iterations = 1)
cv2.imwrite("image_erosion.png",erosion)

# Image closing
closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, None)
cv2.imwrite("image_closing.png",closing)

# image_gray
image_gray = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 9)
cv2.imwrite("image_gray.png",image_gray)

# edge detection
image_edge = cv2.Canny(image_gray,50,100)
image_edge = cv2.dilate(image_edge, None, iterations=1)
image_edge = cv2.erode(image_edge, None, iterations=1)

# contours
contour,hier = cv2.findContours(image_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contour:
    if cv2.contourArea(c) < 5:
        continue
    hull = cv2.convexHull(c)
    cv2.drawContours(image_contorus,[hull],0,(0,0,255),3)
print("{} koloni".format(len(c)))
cv2.imwrite("image_result.png",image_contorus)