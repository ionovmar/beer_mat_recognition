import numpy as np
import argparse
import logging
import cv2
import os


class Detector:
    def __init__(self, filename):
        self.CROP_OFFSET = 3
        self.MIN_AREA = 20000
        self.SCALE_FACTOR = {'x': 1, 'y': 1}
        self.filename = filename
        self.image = self.get_image()

    def get_image(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Get the path to the 'scans' directory using the current file's location
        scans_directory = os.path.join(script_dir, '..', 'data/scans')

        # Specify the path to input file
        tif_file_path = scans_directory + "/" + f"{self.filename}.tif"

        # Read the .tif file
        image = cv2.imread(tif_file_path, cv2.IMREAD_UNCHANGED)

        # Check if the image was successfully loaded
        if image is not None:
            scaled_image = cv2.resize(image, None, fx=self.SCALE_FACTOR['x'], fy=self.SCALE_FACTOR['y'])
            return scaled_image
        else:
            logging.error(f"Unable to read the .tif file at {tif_file_path}")
            raise FileExistsError

    def save_image(self, image):
        # Get the path to the 'scans' directory using the current file's location
        output_directory = os.path.join(os.getcwd(), '..',  'data/outputs')
        # Specify the path to output file
        path_4_saving = output_directory + "/" + f"{self.filename}.png"
        cv2.imwrite(path_4_saving, image)

    def image_preprocessing(self):  
        # Provide morphological operations and find edges
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blurred, 10, 100)

        # Define a (3, 3) structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Apply the dilation operation to the edged image
        dilate = cv2.dilate(edged, kernel, iterations=1)
        return dilate

    def get_contours(self):
        dilate = self.image_preprocessing()
        # Contours detection
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_copy = self.image.copy()
        borders = []
        # Contours drawing
        for cnt in contours:
            # Get the 4 points of the bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > self.MIN_AREA:
                # Draw a straight rectangle with the points
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                borders.append([x, y, w, h])

        return image_copy, contours, borders

    def crop(self, image, borders):
        mats = []
        for border in borders:
            # Cropping an image
            x, y, w, h = border
            if w * h > self.MIN_AREA:
                cropped_image = image[y - self.CROP_OFFSET:(y + h + self.CROP_OFFSET),
                                           x - self.CROP_OFFSET:(x + w + self.CROP_OFFSET)]
                mats.append(cropped_image)
        return mats

    def get_masks(self, contours):
        marks = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100000:
                # Created a new mask and used bitwise_and to select for contours:
                # hull = cv2.convexHull(cnt)
                perimeter = cv2.arcLength(cnt, True)
                approximatedShape = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                mask = np.zeros_like(self.image)
                cv2.drawContours(mask, [approximatedShape], 0, (255, 255, 255), -1)
                masked_image = cv2.bitwise_and(self.image, mask)
                marks.append(masked_image)
                self.show_image(mask)
        return marks

    def show_image(self, image, window_name='Result'):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='detector',
        description='This program applies image processing and contour '
                    'detection operations,and then displays and saves '
                    'the processed image with drawn bounding rectangles '
                    'around detected shapes')

    parser.add_argument('filename')  # positional argument
    parser.add_argument('-v', '--vis', help='show result', action='store_true')  # option that takes a value
    parser.add_argument('-s', '--save', help='save result', action='store_true')  # option that takes a value
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=f"%(levelname)-8s: %(message)s")

    detector = Detector(args.filename)
    result_image, contours, borders = detector.get_contours()
    detector.show_image(result_image, 'Contours')
    masks = detector.get_masks(contours)
    mats = []
    for mask, border in zip(masks, borders):
        mat = np.array(detector.crop(mask, [border]))
        for y in range(len(mat[0])):
            for x in range(len(mat[0][y])):
                if sum(mat[0][y][x]) == 0:
                    mat[0][y][x] = [255, 255, 255]

        mats.append(mat[0])
        detector.show_image(mat[0], window_name='Contours')

    logging.info(f'Summery: \nNumber of contours: {len(contours)}\n'
                 f'Numer of borders: {len(borders)} \n'
                 f'Number of  masks: {len(masks)}\n'
                 f'Final number of mat: {len(mats)}')

    if args.vis:
        detector.show_image(result_image)

    if args.save:
        detector.save_image(result_image)
