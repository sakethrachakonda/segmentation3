'''import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


gc2 = Image.open("E:\DS Assignment\seg2.jpg")
gc2_np = np.array(gc2)
img_pil = Image.fromarray(gc2_np)
img_pil.show()


gray = cv2.cvtColor(gc2_np, cv2.COLOR_BGR2GRAY)


_,thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)


mask = thresholded.astype('uint8')


segmented_img = gc2_np * (mask[:, :, np.newaxis] // 255)


image_with_alpha = np.copy(gc2_np)
image_with_alpha = cv2.cvtColor(image_with_alpha, cv2.COLOR_BGR2BGRA)


image_with_alpha[mask == 0, 3] = 0  
plt.subplot(1, 2, 1)
plt.title("Segmented Image with Transparent BG")
plt.imshow(cv2.cvtColor(image_with_alpha, cv2.COLOR_BGRA2RGBA))  
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(gc2_np, cv2.COLOR_BGR2RGB))  
plt.title("Original Image")
plt.axis("off")

plt.show()'''

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











