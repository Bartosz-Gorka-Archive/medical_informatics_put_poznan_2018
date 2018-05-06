# CORE
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage import filters  # Frangi filter
from matplotlib import pyplot as plt  # Optional - for show part result only

# GUI
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt


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
        wr = Writer()
        wr.save_mask('org', self.picture)
        wr.save_mask('gray', cv2.cvtColor(self.picture, cv2.COLOR_BGR2GRAY))
        img = np.int16(self.picture.copy())
        img[:, :, 0] = 0
        img[:, :, 2] = 0

        wr.save_mask('green', img)

        contrast = 35
        brightness = 90
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        wr.save_mask('contrast_blue', img)
        wr.save_mask('contrast_gray', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def binary_response(self, image):
        threshold = image.mean() * 1.35
        image[image > threshold] = 255
        image[image <= threshold] = 0

        if self.debug:
            print('Threshold =>', threshold)

        return image

    def make_recognition(self):
        wr = Writer()
        gray = self.cut_green_channel_with_contrast()
        wr.save_mask('1_cut_green', gray)
        canny_edges = cv2.Canny(gray, 150, 255, apertureSize=5, L2gradient=True)
        wr.save_mask('2_canny', canny_edges)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilate = cv2.dilate(canny_edges, kernel)
        wr.save_mask('3_dilate', dilate)

        median = cv2.medianBlur(dilate, 5)
        wr.save_mask('4_median_blur', median)

        # pic = filters.frangi(median) # TODO it consumes to much time
        # wr.save_mask('5_frangi', pic)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilate = cv2.dilate(median, kernel)
        wr.save_mask('6_dilate', dilate)

        result = self.binary_response(dilate)
        wr.save_mask('7_result', result)
        return result


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
    total_elements = 2_000

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

            counter += 1
            decisions.append(decision)
            hu_moments.append(hu)

        return hu_moments, decisions

    def cosine_similarity(self, hu_moments, vector):
        similarity = []
        for hu_moment in hu_moments:
            # similarity.append(np.abs(np.subtract(hu_moment, vector)).sum())
            similarity.append(np.dot(vector, hu_moment) / (np.linalg.norm(vector) * np.linalg.norm(hu_moment)))
        return similarity

    def make_decision(self, hu_moments, vector, decisions):
        # index = np.argmin(self.cosine_similarity(hu_moments, vector))
        index = np.argmax(self.cosine_similarity(hu_moments, vector))
        return decisions[index]

    def prepare_response(self, hu_moments, decisions):
        image = self.prepare_image()
        response_image = np.zeros_like(image)

        height, width = self.mask.shape
        mask_radius = int((self.mask_size - 1) / 2)
        max_height = height - mask_radius
        max_width = width - mask_radius

        for row in range(mask_radius, max_height):
            for col in range(mask_radius, max_width):
                cut = image[row - mask_radius: row + mask_radius + 1, col - mask_radius: col + mask_radius + 1]
                vector = cv2.HuMoments(cv2.moments(cut)).flatten()
                response_image[row, col] = self.make_decision(hu_moments, vector, decisions)

        return response_image

    def knn_prepare_response(self, classifier):
        image = self.prepare_image()
        response_image = np.zeros_like(image)
        temp = []

        height, width = self.mask.shape
        mask_radius = int((self.mask_size - 1) / 2)
        max_height = height - mask_radius
        max_width = width - mask_radius

        for row in range(mask_radius, max_height + 1):
            vec = []
            for col in range(mask_radius, max_width + 1):
                cut = image[row - mask_radius: row + mask_radius + 1, col - mask_radius: col + mask_radius + 1]
                vec.append(cv2.HuMoments(cv2.moments(cut)).flatten())

            temp.append(classifier.predict(vec))
            print(row)

        response_image[mask_radius:max_height+1, mask_radius:max_width+1] = temp
        return response_image


class GUIWidget(QWidget):
    placeholder_image_path = 'Files/image_not_available.jpg'
    default_width = 280
    default_height = 280
    default_button_width = 200
    default_button_height = 40

    def __init__(self):
        super().__init__()

        # Variables
        self.no = 1
        self.calculate_stats = True
        self.selected_picture = None
        self.selected_expert_mask = None
        self.selected_classifier_model = None
        self.selected_basic_decisions = None
        self.selected_basic_model = None

        reader = Reader()

        # Labels
        self.label_picture = QLabel(self)
        self.label_expert_mask = QLabel(self)
        self.label_basic_mask = QLabel(self)
        self.label_simple_learn_mask = QLabel(self)
        self.label_rf_mask = QLabel(self)

        # Pictures
        self.picture_original_picture = QLabel(self)
        self.picture_expert_mask = QLabel(self)
        self.picture_basic_mask = QLabel(self)
        self.picture_simple_learn_mask = QLabel(self)
        self.picture_rf_mask = QLabel(self)

        # Buttons
        self.button_clean_calculations = QPushButton('Clean calculations', self)
        self.button_show_stats = QPushButton('Show statistics', self)
        self.button_select_picture = QPushButton('Select picture', self)
        self.button_select_expert_mask = QPushButton('Select expert mask', self)
        self.button_select_basic_model = QPushButton('Select basic model', self)
        self.button_select_basic_decisions = QPushButton('Select decisions model', self)
        self.button_select_classifier_model = QPushButton('Select classifier model', self)
        self.button_run_code = QPushButton('Run calculations', self)

        # Checkboxes
        self.checkbox_calculate_stats = QCheckBox('Calculate statistics', self)

        # Actions
        self.checkbox_calculate_stats.stateChanged.connect(lambda: self.calculate_stats_change())
        self.button_select_picture.clicked.connect(lambda: self.select_picture())
        self.button_select_expert_mask.clicked.connect(lambda: self.select_expert_mask())
        self.button_select_basic_model.clicked.connect(lambda: self.select_basic_model())
        self.button_select_basic_decisions.clicked.connect(lambda: self.select_basic_decisions())
        self.button_select_classifier_model.clicked.connect(lambda: self.select_classifier_model())
        self.button_run_code.clicked.connect(lambda: self.run_code())

        # Init GUI
        self.init_ui()

    def calculate_stats_change(self):
        self.calculate_stats = self.checkbox_calculate_stats.isChecked()

    def run_code(self):
        print('Not implemented yet.')

    def select(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Images (*.png *.tif *.jpg)", options=options)

        return file_name

    def select_numpy(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Numpy files (*.npy)", options=options)

        return file_name

    def select_picture(self):
        file_name = self.select()
        if file_name:
            self.selected_picture = file_name.lower()

    def select_expert_mask(self):
        file_name = self.select()
        if file_name:
            self.selected_expert_mask = file_name.lower()

    def select_basic_model(self):
        file_name = self.select_numpy()
        if file_name:
            self.selected_basic_model = file_name.lower()

    def select_basic_decisions(self):
        file_name = self.select_numpy()
        if file_name:
            self.selected_basic_decisions = file_name.lower()

    def select_classifier_model(self):
        file_name = self.select()
        # TODO correct select
        if file_name:
            self.selected_classifier_model = file_name.lower()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_placeholder(self, picture):
        picture.setPixmap(QPixmap(self.placeholder_image_path).scaled(self.default_height, self.default_width, Qt.KeepAspectRatio))

    def place_picture(self, picture, pos_x, pos_y):
        picture.move(pos_x, pos_y)
        picture.resize(self.default_width, self.default_height)
        self.set_placeholder(picture)

    def place_label(self, label, pos_x, pos_y, text):
        label.setText(text)
        label.move(pos_x, pos_y)

    def place_checkbox(self, checkbox, is_checked, pos_x, pos_y):
        checkbox.move(pos_x, pos_y)
        checkbox.setChecked(is_checked)

    def place_button(self, button, pos_x, pos_y):
        button.move(pos_x, pos_y)
        button.resize(self.default_button_width, self.default_button_height)

    def init_ui(self):
        # Size, position, title
        self.setFixedSize(900, 640)
        self.center()
        self.setWindowTitle('Vessels recognition')

        # Pictures
        self.place_picture(self.picture_original_picture, 15, 30)
        self.place_picture(self.picture_expert_mask, 310, 30)
        self.place_picture(self.picture_basic_mask, 15, 350)
        self.place_picture(self.picture_simple_learn_mask, 310, 350)
        self.place_picture(self.picture_rf_mask, 605, 350)

        # Labels
        self.place_label(self.label_picture, 25, 10, 'Picture')
        self.place_label(self.label_expert_mask, 320, 10, 'Expert mask')
        self.place_label(self.label_basic_mask, 25, 330, 'Basic mask')
        self.place_label(self.label_simple_learn_mask, 320, 330, 'kNN mask')
        self.place_label(self.label_rf_mask, 615, 330, 'Random Forest mask')

        # Buttons
        self.place_button(self.button_select_picture, 650, 30)
        self.place_button(self.button_select_expert_mask, 650, 60)
        self.place_button(self.button_select_basic_model, 650, 90)
        self.place_button(self.button_select_basic_decisions, 650, 120)
        self.place_button(self.button_select_classifier_model, 650, 150)
        self.place_button(self.button_run_code, 650, 180)
        self.place_button(self.button_show_stats, 650, 260)
        self.place_button(self.button_clean_calculations, 650, 290)

        # Checkboxes
        self.place_checkbox(self.checkbox_calculate_stats, self.calculate_stats, 655, 240)

        # Show UI
        self.show()


def main():
    app = QApplication(sys.argv)
    ex = GUIWidget()
    sys.exit(app.exec_())

    # file_name = '01_h'
    # reader = Reader(file_name)
    # writer = Writer()
    #
    # original_image = reader.read_picture()
    # mask = reader.read_mask()
    # expert_mask = reader.read_expert_mask()

    #####################
    #    Recognition    #
    #####################
    # recognition = Recognition(original_image, mask, expert_mask)
    # own_mask = recognition.make_recognition()
    # writer.save_mask(file_name, own_mask)
    # statistics = Statistics()
    # stats = statistics.statistics(expert_mask, own_mask)
    # for (key, val) in stats.items():
    #     print(f'{key} => {val:.{5}f}')

    #####################
    #   SimpleLearner   #
    #####################
    # learner = SimpleLearner(expert_mask, mask, original_image)
    # hu_moments, decisions = learner.learn()
    # writer.save_hu_moments(hu_moments, decisions)
    # r_hu, r_d = reader.read_hu_moments()
    # img = learner.prepare_response(r_hu, r_d)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    #####################
    #   SciKit -> kNN   #
    #####################
    # near_neighbors = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    # near_neighbors.fit(hu_moments, decisions)
    # resp_image = learner.knn_prepare_response(near_neighbors)
    # plt.imshow(resp_image, cmap='gray')
    # plt.show()

    #####################
    #   Random Forest   #
    #####################
    # TODO


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
# * Make binary response - analytics all pixels and check decision from Learner [DONE]
# * KNeighborsClassifier [DONE]
# * PyQT GUI [DONE]
# * Enable make actions from GUI [DONE]

# TODO LIST
# * RandomForest Classifier
# * Cross validation
# * Machine Learning with SciKit

# OPTIONAL
# * Cut image from channel - use mask to reduce extra edges around the eye
# * Speed up prepare response from own Classifier
