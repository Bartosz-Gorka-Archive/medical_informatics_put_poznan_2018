import os
import cv2
import numpy as np
from skimage import filters
from matplotlib import pyplot as plt  # Optional - for show part result only


class Reader:
    pictures_path = 'Files/pictures'
    mask_path = 'Files/masks'
    expert_result_path = 'Files/expert_results'
    picture_ext = '.jpg'
    details_ext = '.tif'

    def __init__(self, file_name):
        self.file_name = file_name

    def read_picture(self):
        path = os.path.join(self.pictures_path, self.file_name) + self.picture_ext
        return cv2.imread(path)

    def read_expert_mask(self):
        path = os.path.join(self.expert_result_path, self.file_name) + self.details_ext
        return cv2.imread(path)

    def read_mask(self):
        path = os.path.join(self.mask_path, self.file_name) + '_mask' + self.details_ext
        return cv2.imread(path)


class Writer:
    clean_directory = False
    result_path = 'Results'

    def __init__(self, clean_dir=False):
        self.clear_result_directory()
        self.create_result_directory()
        self.clean_directory = clean_dir

    def create_result_directory(self):
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path, 0o755)

    def clear_result_directory(self):
        if self.clean_directory:
            os.rmdir(self.result_path)

    def save_mask(self, picture):
        pass


class Recognition:
    debug = True

    def __init__(self, picture, mask, expert_mask, debug=True):
        self.picture = picture.copy()
        self.mask = mask.copy()
        self.expert_mask = expert_mask.copy()
        self.debug = debug

    def cut_green_channel_with_contrast(self):
        img = np.int16(self.picture)
        img[:, :, 0] = 0
        img[:, :, 2] = 0

        contrast = 15
        brightness = 50
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def binary_response(self, image):
        threshold = image.mean() * 1.35
        image[image > threshold] = 255
        image[image <= threshold] = 0

        if self.debug:
            print('Threshold =>', threshold)

        return image

    def make_recognition(self):
        gray = self.cut_green_channel_with_contrast()
        canny_edges = cv2.Canny(gray, 150, 255, apertureSize=5, L2gradient=True)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilate = cv2.dilate(canny_edges, kernel)

        median = cv2.medianBlur(dilate, 5)

        # pic = filters.frangi(median) # TODO it consumes to much time

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilate = cv2.dilate(median, kernel)

        return self.binary_response(dilate)


def main():
    file_name = '01_h'
    reader = Reader(file_name)
    writer = Writer()

    original_image = reader.read_picture()
    mask = reader.read_mask()
    expert_mask = reader.read_expert_mask()

    recognition = Recognition(original_image, mask, expert_mask)
    mask = recognition.make_recognition()
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    writer.save_mask(mask)



if __name__ == "__main__":
    main()

# DONE LIST
# * Read image [DONE]
# * Read expert mask [DONE]
# * Select only green channel from image [DONE]
# * Dilatation [DONE]
# * MedianBlur [DONE]
# * Frangi filter to prepare continuous edges [DONE]
# * Prepare binary response (0/1 - vessel or not) as own mask [DONE]

# TODO LIST
# * Save result as picture
# * Compare own mask with expert mask and calculate statistics
# * Machine Learning with SciKit

# OPTIONAL
# * Cut image from channel - use mask to reduce extra edges around the eye
