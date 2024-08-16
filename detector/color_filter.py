import cv2
import numpy as np
import os


# Function to resize the image
def resize_image(image, width=None, height=None):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Calculate the aspect ratio and resize the image
    if width is not None:
        ratio = width / float(w)
        dimensions = (width, int(h * ratio))
    else:
        ratio = height / float(h)
        dimensions = (int(w * ratio), height)

    # Perform the resizing of the image
    resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return resized

if __name__ == '__main__':
    def nothing(*arg):
        pass

# Get the path to the 'scans' directory using the current file's location
script_directory = os.getcwd()
scans_directory = os.path.join(script_directory, '..', 'data/scans')
output_directory = os.path.join(script_directory, '..', 'data/outputs')

# Specify the path to your .tif file
file = '4bl'

tif_file_path = scans_directory + "/" + f"{file}.tif"
filename_4_saving = output_directory + "/" + f"{file}.png"

# Read the .tif file
image = cv2.imread(tif_file_path, cv2.IMREAD_UNCHANGED)
# Define the desired width or height (set according to your screen size)
desired_width = 800  # You can adjust this value
desired_height = None  # If you prefer to define height, set width to None

# Resize the image
resized_image = resize_image(image, width=desired_width, height=desired_height)

cv2.namedWindow("result")
cv2.namedWindow("settings")

cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
crange = [0, 0, 0, 0, 0, 0]

while True:
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')

    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    thresh = cv2.inRange(hsv, h_min, h_max)

    cv2.imshow('result', thresh)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

# cap.release()
cv2.destroyAllWindows()