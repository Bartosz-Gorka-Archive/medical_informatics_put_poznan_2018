import os
import cv2
from matplotlib import pyplot as plt


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


class Recognition:
    def __init__(self, picture, mask, expert_mask):
        self.picture = picture.copy()
        self.mask = mask.copy()
        self.expert_mask = expert_mask.copy()

    def cut_green_channel(self):
        b, g, r = cv2.split(self.picture)
        return g


def main():
    file_name = '01_h'
    reader = Reader(file_name)

    original_image = reader.read_picture()
    mask = reader.read_mask()
    expert_mask = reader.read_expert_mask()

    recognition = Recognition(original_image, mask, expert_mask)
    green_channel = recognition.cut_green_channel()
    plt.imshow(green_channel, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()

# DONE LIST
# * Read image [DONE]
# * Read expert mask [DONE]
# * Select only green channel from image [DONE]

# TODO LIST
# * Dilatation
# * MedianBlur
# * Save result as picture
# * Machine Learning with SciKit
