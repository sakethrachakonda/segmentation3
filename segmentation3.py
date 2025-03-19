
import cv2
import numpy as np

# Load the leaf image
image = cv2.imread(r"E:\DS Assignment\seg2.jpg")

# Resize image for consistent processing (optional)
image = cv2.resize(image, (600, 400))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 100, 200)  # Adjust threshold values based on image quality

# Display the original and edge-detected images
cv2.imshow("Original Image", image)
cv2.imshow("Canny Edge Detection", edges)

# Save the output
cv2.imwrite("detected_edges.jpg", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()











