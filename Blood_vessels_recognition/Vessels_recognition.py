import os
import cv2
import numpy as np
from skimage import filters
from matplotlib import pyplot as plt  # Optional - for show part result only


class Reader:
    pictures_path = 'Files/pictures'
    mask_path = 'Files/masks'
    expert_result_path = 'Files/expert_results'
    classifier_path = 'Classifier'
    classifier_name = 'classifier'
    picture_ext = '.jpg'
    details_ext = '.tif'

    def __init__(self, file_name):
        self.file_name = file_name

    def read_picture(self):
        path = os.path.join(self.pictures_path, self.file_name + self.picture_ext)
        return cv2.imread(path)

    def read_expert_mask(self):
        path = os.path.join(self.expert_result_path, self.file_name + self.details_ext)
        image = cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def read_mask(self):
        path = os.path.join(self.mask_path, self.file_name + '_mask' + self.details_ext)
        image = cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def read_hu_moments(self):
        name = os.path.join(self.classifier_path, self.classifier_name)
        hu_moments = np.load(name + '_model.npy')
        decisions = np.load(name + '_decision.npy')
        return hu_moments, decisions


class Writer:
    clean_directory = True
    result_path = 'Results'
    picture_ext = '.jpg'
    classifier_path = 'Classifier'
    classifier_name = 'classifier'

    def __init__(self, clean_dir=False):
        self.clear_result_directory()
        self.create_result_directory()
        self.create_classifier_directory()
        self.clean_directory = clean_dir

    def create_result_directory(self):
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path, 0o755)

    def create_classifier_directory(self):
        if not os.path.exists(self.classifier_path):
            os.mkdir(self.classifier_path, 0o755)

    def clear_result_directory(self):
        if self.clean_directory:
            files_to_delete = [os.path.join(self.result_path, file) for file in os.listdir(self.result_path)]
            for file in files_to_delete:
                os.remove(file)

    def save_mask(self, file_name, picture):
        name = os.path.join(self.result_path, file_name + '_mask' + self.picture_ext)
        cv2.imwrite(name, picture)

    def save_hu_moments(self, hu_moments, decisions):
        name = os.path.join(self.classifier_path, self.classifier_name)
        np.save(name + '_model.npy', hu_moments)
        np.save(name + '_decision.npy', decisions)


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


class Statistics:
    def __init__(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

    def compare_masks(self, expert_mask, own_mask):
        for expert_row, own_row in zip(expert_mask, own_mask):
            for expert_cell, own_cell in zip(expert_row, own_row):
                if expert_cell == 0:
                    if own_cell == 0:
                        self.true_negative += 1
                    else:
                        self.false_positive += 1
                else:
                    if own_cell == 0:
                        self.false_negative += 1
                    else:
                        self.true_positive += 1

        return self.true_positive, self.false_positive, self.false_negative, self.true_negative

    def sensitivity(self):
        return self.true_positive / (self.true_positive + self.false_negative)

    def specificity(self):
        return self.true_negative / (self.false_positive + self.true_negative)

    def false_positive_rate(self):
        return self.false_positive / (self.false_positive + self.true_negative)

    def false_discovery_rate(self):
        return self.false_positive / (self.false_positive + self.true_positive)

    def accuracy(self):
        positive = self.true_positive + self.true_negative
        negative = self.false_positive + self.false_negative
        return positive / (positive + negative)

    def positive_predictive_value(self):
        return self.true_positive / (self.true_positive + self.false_positive)

    def negative_predictive_value(self):
        return self.true_negative / (self.true_negative + self.false_negative)

    def statistics(self, expert_mask, own_mask):
        self.compare_masks(expert_mask, own_mask)
        stats = {
            'sensitivity': self.sensitivity(),
            'specificity': self.specificity(),
            'false positive rate': self.false_positive_rate(),
            'false discovery rate': self.false_discovery_rate(),
            'accuracy': self.accuracy(),
            'positive predictive value': self.positive_predictive_value(),
            'negative predictive value': self.negative_predictive_value()
        }
        return stats


class SimpleLearner:
    mask_size = 5
    total_elements = 1_00

    def __init__(self, expert_mask, mask, picture):
        self.expert_mask = expert_mask.copy()
        self.mask = mask.copy()
        self.picture = picture.copy()

    def prepare_image(self):
        recognition = Recognition(self.picture, self.mask, self.picture)
        return recognition.cut_green_channel_with_contrast()

    def learn(self):
        image = self.prepare_image()
        counter = 0
        vessels_counter = 0
        hu_moments = []
        decisions = []

        # Calculate max position, mask radius to min position
        height, width = self.mask.shape
        mask_radius = int((self.mask_size - 1) / 2)
        max_height = height - mask_radius
        max_width = width - mask_radius

        while counter < self.total_elements:
            random_height = np.random.randint(mask_radius, max_height)
            random_width = np.random.randint(mask_radius, max_width)

            cut = image[random_height - mask_radius: random_height + mask_radius + 1, random_width - mask_radius: random_width + mask_radius + 1]
            decision = self.expert_mask[random_height, random_width]
            hu = cv2.HuMoments(cv2.moments(cut)).flatten()

            if decision != 0:
                vessels_counter += 1

            if counter > (self.total_elements * 0.7) and decision == 0:
                continue

            counter += 1
            decisions.append(decision)
            hu_moments.append(hu)

        return hu_moments, decisions

    def cosine_similarity(self, hu_moments, vector):
        similarity = []
        for hu_moment in hu_moments:
            similarity.append(np.dot(vector, hu_moment) / (np.linalg.norm(vector) * np.linalg.norm(hu_moment)))
        return similarity

    def make_decision(self, hu_moments, vector, decisions):
        index = np.argmax(self.cosine_similarity(hu_moments, vector))
        return decisions[index]


def main():
    file_name = '01_h'
    reader = Reader(file_name)
    writer = Writer()

    original_image = reader.read_picture()
    mask = reader.read_mask()
    expert_mask = reader.read_expert_mask()

    # recognition = Recognition(original_image, mask, expert_mask)
    # own_mask = recognition.make_recognition()
    # writer.save_mask(file_name, own_mask)
    # statistics = Statistics()
    # stats = statistics.statistics(expert_mask, own_mask)
    # for (key, val) in stats.items():
    #     print(f'{key} => {val:.{5}f}')
    learner = SimpleLearner(expert_mask, mask, original_image)
    hu_moments, decisions = learner.learn()
    writer.save_hu_moments(hu_moments, decisions)
    r_hu, r_d = reader.read_hu_moments()
    reversed_hu_moments = r_hu[:: -1]
    print(learner.make_decision(reversed_hu_moments, hu_moments[3], decisions))


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
# * Save result as picture [DONE]
# * Compare own mask with expert mask [DONE]
# * Calculate statistics [DONE]
# * Calculate Hu moments - picture analytics [DONE]
# * Save Hu moments to file [DONE]
# * Load Hu moments from file [DONE]
# * Calculate similarity from Hu moments and array from Learner [DONE]
# * Make decision [DONE]

# TODO LIST
# * Make binary response - analytics all pixels and check decision from Learner
# * Machine Learning with SciKit

# OPTIONAL
# * Cut image from channel - use mask to reduce extra edges around the eye
