
import cv2
import numpy as np


image = cv2.imread(r"E:\DS Assignment\seg3.jpg")


image = cv2.resize(image, (600, 400))


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#  Canny edge detection
edges = cv2.Canny(blurred, 100, 200)  # Adjust threshold values based on image quality


cv2.imshow("Original Image", image)
cv2.imshow("Canny Edge Detection", edges)


cv2.imwrite("detected_edges.jpg", edges)

cv2.waitKey(0)#key is used to hold the output untill we want
cv2.destroyAllWindows()











