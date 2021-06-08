import numpy as np
import cv2

# where you stored image
image_origin = cv2.imread("025.png")
height_origin,width_orig = image_origin.shape[:2]

# Create output image with contours
# image output with contours
image_contorus = image_origin.copy()

# store counted colonies
counter = {}

# this code only detect color blue and white only
# Dump colors variable
colors = ['blue','white']
for color in colors:
    # add image to process
    image_to_process = image_origin.copy()
    
    # Initalize counter
    # this variable will store counted bacteria
    counter[color] = 0
    
    if color == 'blue':
        lower = np.array([ 60, 100,  20])
        upper = np.array([170, 180, 150])
    elif color == 'white':
        # invert image colors
        image_to_process = (255-image_to_process)
        lower = np.array([ 50,  50,  40])
        upper = np.array([100, 120,  80])

    # find color from image with specified boundaries
    image_mask = cv2.inRange(image_to_process,lower,upper)
    image_res = cv2.bitwise_and(image_to_process,image_to_process, mask=image_mask)

    # load image and convert to grayscale
    # and change image to blur
    image_gray = cv2.cvtColor(image_res,cv2.COLOR_BGR2GRAY)
    # change image to blur
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    # edge detection
    image_edge = cv2.Canny(image_gray,50,100)
    image_edge = cv2.dilate(image_edge, None, iterations=1)
    image_edge = cv2.erode(image_edge, None, iterations=1)
    
    # detect contours
    contour,hier = cv2.findContours(image_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contour:
        if cv2.contourArea(c) < 5:
            continue
        hull = cv2.convexHull(c)
        if color == "blue":
            # Print contour in image = red
            cv2.drawContours(image_contorus,[hull],0,(0,0,255),4)
        elif color == 'white':
            # Print contour in image = green
            cv2.drawContours(image_contorus,[hull],0,(0,255,0),4)
        counter[color] +=1
        cv2.putText(image_contorus, "{:.0f}".format(cv2.contourArea(c)), (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    print("{} {} colonies".format(counter[color],color))
cv2.imwrite("image_result.png",image_contorus)