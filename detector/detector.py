import numpy as np
import argparse
import logging
import cv2
import os


class Detector:
    def __init__(self, filename):
        self.CROP_OFFSET = 10
        self.MIN_AREA = 10000
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

        edges = cv2.Canny(gray, 100, 200, 9, L2gradient=False)

        # Dilation for circular edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilate = cv2.dilate(edges, kernel, iterations=7)

        return dilate

    def get_contours(self):
        dilate = self.image_preprocessing()

        # Contours detection
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]

        # Filter hierarchy based on hierarchy, take only the outer contours with large area
        hierarchy_new = {}
        new_index = 0
        for i, (next, prev, child, parent) in enumerate(hierarchy):
            if parent == -1 and cv2.contourArea(contours[i]) > self.MIN_AREA:
                hierarchy_new[i] = [new_index, next, prev, child]
                new_index += 1

        for original_index, (new_index, next, prev, child) in hierarchy_new.items():
            if next in hierarchy_new:
                hierarchy_new[original_index][1] = hierarchy_new[next][0]
            if prev in hierarchy_new:
                hierarchy_new[original_index][2] = hierarchy_new[prev][0]
            if child in hierarchy_new:
                hierarchy_new[original_index][3] = hierarchy_new[child][0]

        # Update contours to only keep the contours that are not holes
        contours = [contours[i] for i in range(len(contours)) if
                    hierarchy[i][3] == -1 and cv2.contourArea(contours[i]) > 10000]

        # Update hierarchy_new keys to new_index
        hierarchy_new = {new_index: (next, prev, child) for _, (new_index, next, prev, child) in hierarchy_new.items()}

        # For neighbors, select the outer contour
        for new_index, (next, prev, child) in hierarchy_new.items():
            if next in hierarchy_new and hierarchy_new[next] is not None:
                # Compare these two contours and select the larger one
                if cv2.contourArea(contours[new_index]) < cv2.contourArea(contours[next]):
                    contours[new_index] = None
                    hierarchy_new[new_index] = None
            if prev in hierarchy_new and hierarchy_new[prev] is not None:
                # Compare these two contours and select the larger one
                if cv2.contourArea(contours[new_index]) < cv2.contourArea(contours[prev]):
                    contours[new_index] = None
                    hierarchy_new[new_index] = None

        # Filter contours near (not on the) the border of the image
        height, width = dilate.shape
        border_offset = self.CROP_OFFSET
        contours = [contour for contour in contours if contour is not None and not any([point[0][
                                                                                            0] < border_offset or
                                                                                        point[0][
                                                                                            0] > width - border_offset or
                                                                                        point[0][
                                                                                            1] < border_offset or
                                                                                        point[0][
                                                                                            1] > height - border_offset
                                                                                        for point in contour])]
        # Make the contours closed
        contours = [cv2.approxPolyDP(contour, 0.0001 * cv2.arcLength(contour, False), False) for contour in contours]

        image_copy = self.image.copy()

        borders = []
        image_copy = self.image.copy()
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
        masks = []
        for cnt in contours:
            # Mask image - everything outside the contours stays the original color
            mask = cv2.fillPoly(np.ones_like(self.image), contours, (255, 255, 255))

            # Erode the mask to shrink the contours
            kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel for 2-pixel erosion
            eroded_mask = cv2.erode(mask, kernel, iterations=5)

            # Mask image with eroded mask and keep the original color outside the contours
            white_image = np.ones_like(self.image) * 255
            segmented_image = np.where(eroded_mask == 1, white_image, self.image)
            self.show_image(segmented_image)
            masks.append(segmented_image)
        return masks

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

    parser.add_argument('filename', type=str)  # positional argument
    parser.add_argument('-v', '--vis', help='show result', action='store_true', default=True)  # option that takes a value
    parser.add_argument('-s', '--save', help='save result', action='store_true', default=True)  # option that takes a value
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
                 f'Number of borders: {len(borders)} \n'
                 f'Number of  masks: {len(masks)}\n'
                 f'Final number of mat: {len(mats)}')

    if args.vis:
        detector.show_image(result_image)

    if args.save:
        detector.save_image(result_image)
