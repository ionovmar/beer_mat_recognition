import numpy as np
import argparse
import logging
import cv2
import os

CROP_OFFSET = 3
MIN_AREA = 200000
MAX_AREA = 1000000

# define the alpha and beta
alpha = 0.92    # Contrast control
beta = 10       # Brightness control


def show(image, name='Result'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def set_lim(dir_name):
    if os.path.isfile(dir_name):
        label = os.path.basename(dir_name)
    else:
        label = os.path.dirname(dir_name)
    logging.debug(f"Label {label}")
    if 'green' in label:
        # GREEN
        hsv_min = np.array((56, 66, 10), np.uint8)
        hsv_max = np.array((100, 255, 188), np.uint8)
        logging.info(f'Set hsv lim for green')
    elif 'blue' in label:
        # BLUE
        hsv_min = np.array((28, 160, 73), np.uint8)
        hsv_max = np.array((139, 255, 255), np.uint8)
        logging.info(f'Set hsv lim for blue')
    elif 'red' in label:
        # RED
        hsv_min = np.array((0, 166, 27), np.uint8)
        hsv_max = np.array((180, 255, 255), np.uint8)
        logging.info(f'Set hsv lim for red')
    elif 'black' in label:
        # BLACK
        hsv_min = np.array((0, 0, 17), np.uint8)
        hsv_max = np.array((179, 115, 76), np.uint8)
        logging.info(f'Set hsv lim for black')
    else:
        # WHITE
        hsv_min = np.array((0, 0, 221), np.uint8)  # 0, 0, 221
        hsv_max = np.array((139, 11, 255), np.uint8)  # 139, 11, 255
        logging.debug(f'Set hsv lim for white')

    return hsv_min, hsv_max


def crop_image(masked_image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    if y <= CROP_OFFSET and x <= CROP_OFFSET:
        cropped_image = masked_image[y:(y + h + CROP_OFFSET), x:(x + w + CROP_OFFSET)]
    elif y <= CROP_OFFSET:
        cropped_image = masked_image[y:(y + h + CROP_OFFSET), x - CROP_OFFSET:(x + w + CROP_OFFSET)]
    elif x <= CROP_OFFSET:
        cropped_image = masked_image[y - CROP_OFFSET:(y + h + CROP_OFFSET), x:(x + w + CROP_OFFSET)]
    else:
        cropped_image = masked_image[y - CROP_OFFSET:(y + h + CROP_OFFSET),
                        x - CROP_OFFSET:(x + w + CROP_OFFSET)]
    return cropped_image


def save_image(image, index, filename, output_dir):
    if filename is None and os.path.isfile(output_dir):
        filename = os.path.basename(output_dir).split('.')[0]
        output_dir = os.path.dirname(output_dir)
    else:
        filename = filename.split('.')[0]
        output_dir = os.path.join(output_dir, 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Specify the path to output file
    cv2.imwrite(os.path.join(output_dir, f"{filename}_{index}.png"), image)


def image_processing(hsv_min, hsv_max, input_dir, filename=None):
    # Read the .tif file
    if filename is None:
        image = cv2.imread(input_dir, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)

    # Apply contrast and brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Apply color filter and invert it
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    inverted_thresh = cv2.bitwise_not(thresh)

    # Find contours in the mask
    mask_contours, _ = cv2.findContours(inverted_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mat_counter = 0
    # Loop over the contours
    for contour in mask_contours:
        area = cv2.contourArea(contour)
        if MIN_AREA < area:
            inverted_thresh_copy = inverted_thresh.copy()
            thresh_copy = thresh.copy()

            # Detect the convex contour
            hull = cv2.convexHull(contour)

            # Convert hull points to the correct format for cv2.fillPoly
            hull_points = np.array(hull, dtype=np.int32)

            # Draw a filled polygon on the mask
            cv2.fillPoly(inverted_thresh_copy, [hull_points], 255)
            cv2.fillPoly(thresh_copy, [hull_points], 255)

            # Apply the mask to the output to retain pixels outside the hull
            final_mask = cv2.bitwise_and(thresh_copy, inverted_thresh_copy)

            # Create a white image of the same size as the original image
            white_image = np.ones_like(image) * 255

            # Apply the mask: where mask is 0, use white_image; where mask is 1, use the original image
            masked_image = np.where(final_mask[:, :, np.newaxis] == 0, white_image, image)

            # Cropping an image
            output_image = crop_image(masked_image, contour)

            save_image(output_image, mat_counter, filename, input_dir)
            mat_counter += 1


def main(input_dir):
    hsv_min, hsv_max = set_lim(input_dir)

    if os.path.isdir(input_dir):
        # Processing all images in the input directory
        for filename in os.listdir(input_dir):
            logging.info(f'Processing file {filename}')
            if os.path.isfile(os.path.join(input_dir, filename)):
                image_processing(hsv_min, hsv_max, input_dir, filename)

    elif os.path.isfile(input_dir):
        logging.info(f'Processing file {input_dir}')
        # Processing one image
        image_processing(hsv_min, hsv_max, input_dir)
    else:
        raise FileExistsError


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format=f"%(levelname)-8s: %(message)s")

    parser = argparse.ArgumentParser(
        prog='detector',
        description='This program applies image processing and contour '
                    'detection operations,and then displays and saves '
                    'the processed image with drawn bounding rectangles '
                    'around detected shapes')

    parser.add_argument('input_dir')  # positional argument
    parser.add_argument('-v', '--vis', help='show result', action='store_true')  # option that takes a value
    args = parser.parse_args()

    main(args.input_dir)





