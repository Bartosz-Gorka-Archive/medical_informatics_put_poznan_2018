# CORE
import os
import cv2
import math
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold, cross_val_score, cross_val_predict

# GUI
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


class Reader:
    @staticmethod
    def read_picture(path):
        return cv2.imread(path)

    @staticmethod
    def read_gray_picture(path):
        image = cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def read_numpy_array(path):
        np_array = np.load(path)
        return np_array


class Writer:
    iteration = 0
    result_path = 'Results'
    picture_ext = '.jpg'
    classifier_path = 'Classifier'
    classifier_name = 'classifier'

    def __init__(self, iteration=0):
        self.create_result_directory()
        self.create_classifier_directory()
        self.iteration = iteration

    def create_result_directory(self):
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path, 0o755)

    def create_classifier_directory(self):
        if not os.path.exists(self.classifier_path):
            os.mkdir(self.classifier_path, 0o755)

    def save_mask(self, file_name, picture):
        path = os.path.join(self.result_path, file_name + '_' + str(self.iteration) + self.picture_ext)
        cv2.imwrite(path, picture)
        return path

    def save_numpy_array(self, file_name, numpy_array):
        path = os.path.join(self.classifier_path, file_name + '_' + str(self.iteration))
        np.save(path + '.npy', numpy_array)
        return path


class Recognition:
    def __init__(self, picture, expert_mask):
        self.picture = picture.copy()
        self.expert_mask = expert_mask.copy()

    def cut_green_channel_with_contrast(self):
        img = np.int16(self.picture.copy())
        img[:, :, 0] = 0
        img[:, :, 2] = 0

        contrast = 35
        brightness = 90
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def binary_response(image):
        threshold = image.mean() * 1.35
        image[image > threshold] = 255
        image[image <= threshold] = 0

        print('Threshold =>', threshold)
        return image

    def make_recognition(self):
        gray = self.cut_green_channel_with_contrast()
        canny_edges = cv2.Canny(gray, 150, 255, apertureSize=5, L2gradient=True)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilate = cv2.dilate(canny_edges, kernel)

        median = cv2.medianBlur(dilate, 5)

        # frangi_result = filters.frangi(median) # TODO it consumes to much time and prepare invalid result in code

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilate = cv2.dilate(median, kernel)

        result = self.binary_response(dilate)
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

    def matthews_correlation(self):
        return (self.true_positive * self.true_negative - self.false_negative * self.false_positive) / math.sqrt((self.true_positive + self.false_positive) * (self.true_positive + self.false_negative) * (self.true_negative + self.false_positive) * (self.true_negative + self.false_negative))

    def statistics(self, expert_mask, own_mask):
        self.compare_masks(expert_mask, own_mask)
        stats = {
            'sensitivity': self.sensitivity(),
            'specificity': self.specificity(),
            'false positive rate': self.false_positive_rate(),
            'false discovery rate': self.false_discovery_rate(),
            'accuracy': self.accuracy(),
            'positive predictive value': self.positive_predictive_value(),
            'negative predictive value': self.negative_predictive_value(),
            'Matthews correlation coefficient': self.matthews_correlation()
        }
        return stats


class SimpleLearner:
    mask_size = 21
    total_elements = 100_000
    slice_size = 10
    median_blur_size = 5
    dilate_mask_size = 3

    def __init__(self, expert_mask, picture):
        self.expert_mask = expert_mask.copy()
        self.picture = picture.copy()

    def prepare_image(self):
        recognition = Recognition(self.picture, self.picture)
        return recognition.cut_green_channel_with_contrast()

    def learn(self):
        image = self.prepare_image()
        counter = 0
        cut_masks = []
        decisions = []

        # Calculate max position, mask radius to min position
        height, width = self.expert_mask.shape
        mask_radius = int((self.mask_size - 1) / 2)
        max_height = height - mask_radius
        max_width = width - mask_radius

        while counter < self.total_elements:
            random_height = np.random.randint(mask_radius, max_height)
            random_width = np.random.randint(mask_radius, max_width)

            cut = image[random_height - mask_radius: random_height + mask_radius + 1, random_width - mask_radius: random_width + mask_radius + 1]
            decision = self.expert_mask[random_height, random_width]

            counter += 1
            decisions.append(decision)
            cut_masks.append(cut.flatten())

        return cut_masks, decisions

    @staticmethod
    def cosine_similarity(hu_moments, vector):
        similarity = []
        for hu_moment in hu_moments:
            similarity.append(np.dot(vector, hu_moment) / (np.linalg.norm(vector) * np.linalg.norm(hu_moment)))
        return similarity

    def make_decision(self, hu_moments, vector, decisions):
        index = np.argmax(self.cosine_similarity(hu_moments, vector))
        return decisions[index]

    def basic_prepare_response(self, hu_moments, decisions):
        image = self.prepare_image()
        response_image = np.zeros_like(image)

        height, width = self.expert_mask.shape
        mask_radius = int((self.mask_size - 1) / 2)
        max_height = height - mask_radius
        max_width = width - mask_radius

        for row in range(mask_radius, max_height):
            for col in range(mask_radius, max_width):
                cut = image[row - mask_radius: row + mask_radius + 1, col - mask_radius: col + mask_radius + 1]
                response_image[row, col] = self.make_decision(hu_moments, cut.flatten(), decisions)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_mask_size, self.dilate_mask_size))
        response_image = cv2.dilate(response_image, kernel)
        response_image = cv2.medianBlur(response_image, ksize=self.median_blur_size)
        return response_image

    def classifier_prepare_response(self, classifier, filtering):
        image = self.prepare_image()
        response_image = np.zeros_like(image)
        temp = []

        height, width = self.expert_mask.shape
        mask_radius = int((self.mask_size - 1) / 2)
        max_height = height - mask_radius
        max_width = width - mask_radius

        for row in range(mask_radius, max_height):
            vec = []
            for col in range(mask_radius, max_width):
                cut = image[row - mask_radius: row + mask_radius + 1, col - mask_radius: col + mask_radius + 1]
                vec.append(cut.flatten())

            temp.append(classifier.predict(vec))

        response_image[mask_radius:max_height, mask_radius:max_width] = temp

        if filtering:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_mask_size, self.dilate_mask_size))
            response_image = cv2.dilate(response_image, kernel)
            response_image = cv2.medianBlur(response_image, ksize=self.median_blur_size)

        return response_image

    def cross_validation(self, elements, decisions):
        total_elements = len(elements)
        random_elements_num = int(total_elements * (self.slice_size / 100))
        results = []
        scores = []

        # Loop to generate validations
        for i in range(int(100 / self.slice_size)):
            classifier = RandomForestClassifier()
            score = 0
            indexes = np.random.choice(total_elements, random_elements_num, replace=False)
            mask = np.ones_like(decisions, dtype=bool)
            mask[~indexes] = False

            random_elements_test = []
            random_elements_learn = []
            random_decisions_test = []
            random_decisions_learn = []

            # Split to learn & test set
            for index, status in enumerate(mask):
                if status:
                    random_decisions_learn.append(decisions[index])
                    random_elements_learn.append(elements[index])
                else:
                    random_decisions_test.append(decisions[index])
                    random_elements_test.append(elements[index])

            # Fit classifier
            classifier.fit(random_elements_learn, random_decisions_learn)

            # Check classifier stats
            predicted = classifier.predict(random_elements_test)
            for index, value in enumerate(predicted):
                if value == random_decisions_test[index]:
                    score += 1

            print('Iteration', str(i+1), '->', str(score))
            scores.append(score)
            results.append(classifier)
        print('Mean', np.mean(scores))
        print('Min', np.min(scores))
        print('Max', np.max(scores))
        print('Median', np.median(scores))
        return results[np.argmax(scores)]

    def sklearn_cross_validation(self, classifier, elements, decisions):
        k_fold = KFold(n_splits=self.slice_size)
        validate_details = cross_validate(classifier, elements, decisions, cv=k_fold)
        print(validate_details)

        scores = cross_val_score(classifier, elements, decisions, cv=k_fold)
        print('Cross - validated scores:', scores)

        predictions = cross_val_predict(classifier, elements, decisions, cv=k_fold)
        accuracy = metrics.r2_score(decisions, predictions)
        print('Cross - Predicted Accuracy:', accuracy)

        return classifier


class GUIWidget(QWidget):
    placeholder_image_path = 'Files/image_not_available.jpg'
    default_width = 280
    default_height = 280
    default_button_width = 200
    default_button_height = 40
    show_stats = False

    def __init__(self):
        super().__init__()

        # Variables
        self.no = 0
        self.calculate_stats = True
        self.selected_picture = None
        self.selected_expert_mask = None
        self.selected_classifier_model_knn = None
        self.selected_classifier_model_rf = None
        self.selected_basic_decisions = None
        self.selected_basic_model = None

        self.reader = Reader()

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
        self.button_select_classifier_model_knn = QPushButton('Select classifier model kNN', self)
        self.button_select_classifier_model_rf = QPushButton('Select classifier model RF', self)
        self.button_run_code = QPushButton('Run calculations', self)

        # Checkboxes
        self.checkbox_calculate_stats = QCheckBox('Calculate statistics', self)

        # Actions
        self.checkbox_calculate_stats.stateChanged.connect(lambda: self.calculate_stats_change())
        self.button_select_picture.clicked.connect(lambda: self.select_picture())
        self.button_select_expert_mask.clicked.connect(lambda: self.select_expert_mask())
        self.button_select_basic_model.clicked.connect(lambda: self.select_basic_model())
        self.button_select_basic_decisions.clicked.connect(lambda: self.select_basic_decisions())
        self.button_select_classifier_model_rf.clicked.connect(lambda: self.select_classifier_model_rf())
        self.button_select_classifier_model_knn.clicked.connect(lambda: self.select_classifier_model_knn())
        self.button_show_stats.clicked.connect(lambda: self.toggle_stats())
        self.button_run_code.clicked.connect(lambda: self.run_code())

        # Table - statistics
        self.table = QTableWidget(self)

        # Init GUI
        self.init_ui()

    def calculate_stats_change(self):
        self.calculate_stats = self.checkbox_calculate_stats.isChecked()

    def run_code(self):
        self.no += 1
        writer = Writer(self.no)

        picture = self.reader.read_picture(self.selected_picture)
        self.set_image(self.picture_original_picture, self.selected_picture)

        expert_mask = self.reader.read_gray_picture(self.selected_expert_mask)
        self.set_image(self.picture_expert_mask, self.selected_expert_mask)

        # Basic
        recognition = Recognition(picture, expert_mask)
        own_mask = recognition.make_recognition()
        basic_mask_path = writer.save_mask('basic_mask', own_mask)
        self.set_image(self.picture_basic_mask, basic_mask_path)

        if self.calculate_stats:
            stats = Statistics().statistics(expert_mask, own_mask)
            self.update_table_stats('Basic alg.', stats)

        # kNN
        if self.selected_classifier_model_knn:
            knn_classifier = joblib.load(self.selected_classifier_model_knn)
            knn_learner = SimpleLearner(expert_mask, picture)
            own_mask = knn_learner.classifier_prepare_response(knn_classifier, filtering=True)

            knn_mask_path = writer.save_mask('knn_mask', own_mask)
            self.set_image(self.picture_simple_learn_mask, knn_mask_path)

            if self.calculate_stats:
                stats = Statistics().statistics(expert_mask, own_mask)
                self.update_table_stats('kNN (3) alg.', stats)

        # Random Forest
        if self.selected_classifier_model_rf:
            rf_classifier = joblib.load(self.selected_classifier_model_rf)
            rf_learner = SimpleLearner(expert_mask, picture)
            own_mask = rf_learner.classifier_prepare_response(rf_classifier, filtering=True)

            rf_mask_path = writer.save_mask('rf_mask', own_mask)
            self.set_image(self.picture_rf_mask, rf_mask_path)

            if self.calculate_stats:
                stats = Statistics().statistics(expert_mask, own_mask)
                self.update_table_stats('RandomForest alg.', stats)

    def set_image(self, view, path):
        view.setPixmap(QPixmap(path).scaled(self.default_height, self.default_width, Qt.KeepAspectRatio))

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

    def select_classifier(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Classifier files (*.pkl)", options=options)

        return file_name

    def select_picture(self):
        file_name = self.select()
        if file_name:
            self.selected_picture = file_name.lower()
        else:
            self.selected_picture = None

    def select_expert_mask(self):
        file_name = self.select()
        if file_name:
            self.selected_expert_mask = file_name.lower()
        else:
            self.selected_expert_mask = None

    def select_basic_model(self):
        file_name = self.select_numpy()
        if file_name:
            self.selected_basic_model = file_name.lower()
        else:
            self.selected_basic_model = None

    def select_basic_decisions(self):
        file_name = self.select_numpy()
        if file_name:
            self.selected_basic_decisions = file_name.lower()
        else:
            self.selected_basic_decisions = None

    def select_classifier_model_rf(self):
        file_name = self.select_classifier()
        if file_name:
            self.selected_classifier_model_rf = file_name.lower()
        else:
            self.selected_classifier_model_rf = None

    def select_classifier_model_knn(self):
        file_name = self.select_classifier()
        if file_name:
            self.selected_classifier_model_knn = file_name.lower()
        else:
            self.selected_classifier_model_knn = None

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

    def toggle_stats(self):
        self.table.setHidden(self.show_stats)
        self.show_stats = not self.show_stats

    def update_table_stats(self, name, stats):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        for col, (key, value) in enumerate(stats.items()):
            item = QTableWidgetItem(f'{value:.{5}f}')
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col + 1, item)

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
        self.place_button(self.button_select_classifier_model_knn, 650, 150)
        self.place_button(self.button_select_classifier_model_rf, 650, 180)
        self.place_button(self.button_run_code, 650, 210)
        self.place_button(self.button_show_stats, 650, 290)
        self.place_button(self.button_clean_calculations, 650, 320)
        self.button_clean_calculations.setHidden(True)

        # Checkboxes
        self.place_checkbox(self.checkbox_calculate_stats, self.calculate_stats, 655, 270)

        # Table
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(['Algorithm', 'Sensitivity', 'Specificity', 'FPR', 'PDR', 'Accuracy', 'PPV', 'NPV', 'MCC'])
        self.table.setShowGrid(True)
        self.table.move(10, 10)
        self.table.resize(590, 620)
        self.table.setHidden(True)

        # Show UI
        self.show()


def main():
    app = QApplication(sys.argv)
    ex = GUIWidget()
    sys.exit(app.exec_())

    # file_name = '06'
    # reader = Reader()
    # expert_mask = reader.read_gray_picture(f'Files/expert_results/{file_name}_h.tif')
    #
    # print('File', file_name)
    # print('Random Forest statistics')
    # own_mask = reader.read_gray_picture(f'Results/21_100000/{file_name}_21_rf_100000.jpg')
    # stats = Statistics().statistics(expert_mask, own_mask)
    # for (key, value) in stats.items():
    #     print(key, value)
    #
    # print('\nBasic statistics')
    # own_mask = reader.read_gray_picture(f'Results/21_100000/{file_name}_basic_100000.jpg')
    # stats = Statistics().statistics(expert_mask, own_mask)
    # for (key, value) in stats.items():
    #     print(key, value)


if __name__ == "__main__":
    main()
