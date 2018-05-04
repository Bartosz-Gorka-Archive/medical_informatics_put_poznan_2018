import os
import cv2


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


def main():
    file_name = '01_h'
    reader = Reader(file_name)

    original_image = reader.read_picture()
    print(original_image.shape)

    expert_mask = reader.read_expert_mask()
    print(expert_mask.shape)

    mask = reader.read_mask()
    print(mask.shape)


if __name__ == "__main__":
    main()

# DONE LIST
# * Read image [DONE]
# * Read expert mask [DONE]

# TODO LIST
# * Select only green channel from image
# * Dilatation
# * MedianBlur
# * Save result as picture
# * Machine Learning with SciKit
