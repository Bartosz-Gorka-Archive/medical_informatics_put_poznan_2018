# Core imports
import os
import cv2
import math
import pydicom
import datetime
import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham
from timeit import default_timer as timer
from pydicom.dataset import Dataset, FileDataset

# GUI imports
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt


def read_file(filename):
    # Read file in gray scale
    file = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Check size of original image
    height, width = file.shape
    # Return detail
    return file, height, width


def prepare_circle(file, height, width):
    # Calculation diagonal
    diagonal = int(math.sqrt(math.pow(height, 2) + math.pow(width, 2)))
    # Append empty values outside the image
    image = np.zeros((diagonal, diagonal), dtype=np.uint8)

    # Padding in view
    padding_y = int((diagonal - height) / 2)
    padding_x = int((diagonal - width) / 2)

    # Append original picture
    image[padding_y: padding_y + height, padding_x: padding_x + width] = file

    # Calculation radius
    radius = int(diagonal / 2)

    # Return calculations
    return image, radius


def write_file(filename, image):
    # Result path
    path = "results/" + filename + ".jpg"
    # Prepare blank image with the same size like original image
    normalized_image = np.zeros_like(image)
    # Normalize image (values 0-255)
    cv2.normalize(image, normalized_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Save normalized image
    cv2.imwrite(path, normalized_image)


def calculate_iterations_angle(iterations):
    # We want prepare scan in full angle (360 degrees)
    return 360.0 / iterations


def calculate_angle_between_detectors(no_detectors, scan_angle):
    # List with positions of detectors
    results = list()
    # When have only one detector - make it different
    if no_detectors >= 2:
        # Angle between detectors
        angle = scan_angle / (no_detectors - 1)

        # Iterate indexes and set detectors
        for ind in range(no_detectors):
            results.append(ind * angle)
    else:
        # Move single detector to center of scan angle
        results.append(scan_angle / 2)

    return results


def emitter_position(radius, alpha_in_radians, center_position):
    # Calculate x, y positions of emitter
    x = int(radius * np.cos(alpha_in_radians)) + center_position
    y = int(radius * np.sin(alpha_in_radians)) + center_position
    return x, y


def detectors_position(radius, alpha_in_radians, scan_angle_in_radius, center_position, no_detectors):
    # List with positions of detectors
    detectors = list()

    # add way to calculate positions when passed only 1 detector - eliminate raise error divide by zero.
    original_no_detectors = no_detectors
    if no_detectors == 1:
        no_detectors = 3

    for index in range(no_detectors):
        # Calculate value from formula
        value = np.pi + alpha_in_radians - (scan_angle_in_radius / 2) + (index * (scan_angle_in_radius / (no_detectors - 1)))

        # Calculate x and y position's part
        x = int(radius * np.cos(value)) + center_position
        y = int(radius * np.sin(value)) + center_position

        # Make detector
        detector = (x, y)

        # Add position to list
        detectors.append(detector)

    # Return calculated positions
    if original_no_detectors == 1:
        return list(detectors[1])
    else:
        return detectors


def sum_values_on_line(picture, emitter, detector):
    # Split positions of emitter and detector to x, y
    emitter_x, emitter_y = emitter
    detector_x, detector_y = detector

    # Prepare line from emitter to detector - use Bresenham interpolation
    line = bresenham(emitter_x, emitter_y, detector_x, detector_y)

    # Sum - result
    value = 0.0

    # Picture shape
    x, y = picture.shape

    # Loop to fetch values from picture
    for position_x, position_y in line:
        if position_x < x and position_y < y:
            value += picture[position_x, position_y]

    # Return sum
    return value


def make_sinogram_line(picture, emitter, detectors):
    line = np.zeros(len(detectors))
    for index, selected_detector in enumerate(detectors):
        line[index] = sum_values_on_line(picture, emitter, selected_detector)
    return line


def make_sinogram(picture, radius, no_iterations, scan_angle, no_detectors):
    # Angle per iteration, scan angle in radius
    angle_per_iteration = calculate_iterations_angle(no_iterations)
    scan_angle_in_radius = np.radians(scan_angle)
    # Empty results - already with correct size no_iterations x no_detectors
    result = np.empty((no_iterations, no_detectors))

    # Loop to generate line in sinogram
    for iteration in range(no_iterations):
        # Calculate alpha value in radians
        alpha_in_radians = np.radians(angle_per_iteration * iteration)
        # Calculate positions of emitter and detectors
        emitter = emitter_position(radius, alpha_in_radians, radius)
        detectors = detectors_position(radius, alpha_in_radians, scan_angle_in_radius, radius, no_detectors)
        # Generate single line of sinogram
        line = make_sinogram_line(picture, emitter, detectors)
        # Set value in results array
        result[iteration] = line

    # Normalize result to ensure correct format and values inside array
    cv2.normalize(result, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # Return prepared sinogram
    return result


def reverse_sinogram(mse_enabled, original, picture, sinogram, radius, no_iterations, scan_angle, no_detectors, height, width):
    # Prepare zeros array with the same size like picture
    result = np.zeros_like(picture, dtype=float)
    mse_errors = np.zeros(no_iterations)

    # Angle per iteration, scan angle in radius
    angle_per_iteration = calculate_iterations_angle(no_iterations)
    scan_angle_in_radius = np.radians(scan_angle)

    # Shape
    x, y = picture.shape

    # Loop to place values from sinogram
    for iteration in range(no_iterations):
        # Calculate alpha value in radians
        alpha_in_radians = np.radians(angle_per_iteration * iteration)
        # Calculate positions of emitter and detectors
        emitter_x, emitter_y = emitter_position(radius, alpha_in_radians, radius)
        detectors = detectors_position(radius, alpha_in_radians, scan_angle_in_radius, radius, no_detectors)

        for index, detector in enumerate(detectors):
            # Split position of detector to x, y
            detector_x, detector_y = detector
            # Line between emitter and selected detector
            line = bresenham(emitter_x, emitter_y, detector_x, detector_y)

            for position_x, position_y in line:
                if position_x < x and position_y < y:
                    # Append values to image
                    result[position_x, position_y] += sinogram[iteration, index]

        # Save iteration results
        picture = cut_original_size(result, height, width)
        write_file("cut_" + str(iteration), picture)

        # Calculate MSE for iteration
        if mse_enabled:
            mse_errors[iteration] = calculate_mse(original, picture)

    # Return prepared image with MSE errors
    return result, mse_errors


def cut_original_size(picture, height, width):
    # Calculation diagonal
    diagonal = int(math.sqrt(math.pow(height, 2) + math.pow(width, 2))) + 10

    # Padding in view
    padding_y = int((diagonal - height) / 2)
    padding_x = int((diagonal - width) / 2)

    # Cut original format from picture
    cut_picture = picture[padding_y: padding_y + height, padding_x: padding_x + width].copy()

    # Return cut picture
    return cut_picture


def calculate_max_mse(picture):
    # Fetch picture size
    x, y = picture.shape
    # Max MSE
    max_mse = np.empty(x * y)

    # Iterate over picture
    index = 0
    for row in picture:
        for value in row:
            # Calculate max distance for MSE - if color > 127 use 0, else use 255 as value (the largest distance)
            max_mse[index] = calculate_difference(value, 0 if value > 127 else 255)
            index += 1

    # Return max MSE value
    return math.sqrt(max_mse.mean())


def calculate_difference(original, check):
    # Calculate second power
    return math.pow(original - check, 2)


def calculate_mse(original, result):
    # MSE value
    mse = math.sqrt(np.power(np.subtract(original, result), 2).mean())

    # Return MSE value
    return mse


def prepare_mse_graph(max_mse, mse_errors):
    # Axis x and y values
    axis_x = np.arange(len(mse_errors))
    axis_y = [math.ceil((error / max_mse) * 10_000)/100 for error in mse_errors]

    plt.plot(axis_x, axis_y, color='red', marker='o', linestyle='None', markersize=2)
    plt.title('MSE changes')
    plt.xlabel('Iteration')
    plt.ylabel('MSE [%]')
    plt.ylim([0, 100])
    plt.savefig('results/mse_graph.jpg')
    plt.gcf().clear()


def prepare_ramlak_mask(length):
    # Empty array
    mask = np.empty(length, dtype=float)

    # Nominator constant part
    nominator = (-4 / math.pow(np.pi, 2)) * 255

    # Loop to prepare filter's values
    for index in range(length):
        if index % 2 == 0:
            mask[index] = 0.0
        else:
            mask[index] = nominator / math.pow(index, 2)

    # Set first value
    mask[0] = 255

    # Return prepared mask
    return mask


def make_convolve(sinogram, mask):
    # Filtered sinogram
    filtered = np.empty_like(sinogram)

    # Top used variables
    mask_inf_length = mask.size - 1
    row_inf_length = sinogram[0].size - 1

    # Loop to prepare convolve
    for i, row in enumerate(sinogram):
        for j, value in enumerate(row):
            filtered[i][j] = convolve(j, mask_inf_length, mask, row_inf_length, sinogram[i])

    # Return prepared, filtered sinogram
    return filtered


def convolve(current_position, mask_inf_length, mask, row_inf_length, row):
    # Value in position
    value = 0.0

    # Calculate position
    start = 0 if current_position - mask_inf_length <= 0 else current_position - mask_inf_length
    end = row_inf_length if current_position + mask_inf_length > row_inf_length else current_position + mask_inf_length

    # Calculate current center
    current_center = int((end - start) / 2)

    for index in range(start, end):
        # Add value from sinogram x mask
        value += mask[abs(current_center)] * row[index]
        # Move current selected field in mask
        current_center -= 1

    # Return value in cell
    return value


class Tomography(QWidget):
    def __init__(self):
        super().__init__()

        # Variables
        self.debug = True
        self.mask_percent = 0.20
        self.filter_enable = False
        self.mse_enable = False
        self.selected_file = None
        self.already_show_mse = False
        self.no_detectors = 0
        self.iterations = 0
        self.scan_angle = 0
        self.selected_iteration = 0
        self.patient_name = ""
        self.patient_id = ""
        self.dicom_date = ""
        self.save_as_dicom = False

        # Top used objects
        self.input_detectors = QLineEdit(self)
        self.no_iterations_input = QLineEdit(self)
        self.scan_angle_input = QLineEdit(self)
        self.selected_iteration_input = QLineEdit(self)
        self.mse_graph = QLabel(self)
        self.label_original_picture = QLabel(self)
        self.original_picture_basic = QLabel(self)
        self.label_sinogram_picture = QLabel(self)
        self.sinogram_picture_basic = QLabel(self)
        self.selected_picture_basic = QLabel(self)
        self.label_reverse_picture = QLabel(self)
        self.reverse_picture_basic = QLabel(self)
        self.label_selected_picture = QLabel(self)
        self.dicom_date_input = QDateEdit(self)
        self.patient_name_input = QLineEdit(self)
        self.patient_id_input = QLineEdit(self)
        self.checkbox_filter = QCheckBox('Enable filter', self)
        self.mse_checkbox = QCheckBox('Calculate MSE', self)
        self.save_dicom_checkbox = QCheckBox('Save result as DICOM', self)
        self.button_read_file = QPushButton('Select file', self)
        self.button_run_code = QPushButton('Run tomography', self)
        self.button_run_again = QPushButton('Clear && enable again', self)
        self.button_show_mse = QPushButton('Show MSE graph', self)
        self.button_show_selected = QPushButton('Show selected iteration', self)

        # Actions
        self.checkbox_filter.stateChanged.connect(lambda: self.filter_change())
        self.mse_checkbox.stateChanged.connect(lambda: self.mse_change())
        self.button_read_file.clicked.connect(lambda: self.select_file())
        self.button_run_code.clicked.connect(lambda: self.run_code())
        self.button_show_mse.clicked.connect(lambda: self.show_mse_graph())
        self.button_run_again.clicked.connect(lambda: self.run_again())
        self.button_show_selected.clicked.connect(lambda: self.show_cut_file())
        self.save_dicom_checkbox.stateChanged.connect(lambda: self.save_dicom_change())

        # Init UI
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(880, 680)
        self.center()
        self.setWindowTitle('Tomography simulation')

        # MSE graph
        self.mse_graph.move(10, 80)
        self.mse_graph.resize(640, 480)
        self.mse_graph.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(480, 640, Qt.KeepAspectRatio))
        self.mse_graph.setVisible(False)

        # First image label
        self.label_original_picture.setText('Original picture')
        self.label_original_picture.move(15, 10)

        # First image - original picture
        self.original_picture_basic.move(15, 30)
        self.original_picture_basic.resize(300, 300)
        self.original_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))

        # Sinogram image label
        self.label_sinogram_picture.setText('Sinogram')
        self.label_sinogram_picture.move(335, 10)

        # Sinogram base image
        self.sinogram_picture_basic.move(330, 30)
        self.sinogram_picture_basic.resize(300, 300)
        self.sinogram_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))

        # Reverse image label
        self.label_reverse_picture.setText('Reverse image')
        self.label_reverse_picture.move(15, 340)

        # Reverse image - original picture
        self.reverse_picture_basic.move(15, 360)
        self.reverse_picture_basic.resize(300, 300)
        self.reverse_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))

        # Selected image label
        self.label_selected_picture.setText('Selected step image')
        self.label_selected_picture.move(335, 340)

        # Selected image - original picture
        self.selected_picture_basic.move(330, 360)
        self.selected_picture_basic.resize(300, 300)
        self.selected_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))

        # Settings label
        bold_font = QFont()
        bold_font.setBold(True)
        label_project_settings = QLabel(self)
        label_project_settings.setText('Project settings')
        label_project_settings.move(670, 10)
        label_project_settings.setFont(bold_font)

        # No detectors label
        no_detectors_label = QLabel(self)
        no_detectors_label.setText('Number of detectors')
        no_detectors_label.move(670, 30)

        # No detectors input
        self.input_detectors.setValidator(QIntValidator())
        self.input_detectors.setMaxLength(4)
        self.input_detectors.setPlaceholderText('No. detectors')
        self.input_detectors.setAlignment(Qt.AlignCenter)
        self.input_detectors.resize(180, 24)
        self.input_detectors.setFocus()
        self.input_detectors.move(670, 50)

        # No detectors label
        no_iterations_label = QLabel(self)
        no_iterations_label.setText('Number of iterations')
        no_iterations_label.move(670, 80)

        # No iterations input
        self.no_iterations_input.setValidator(QIntValidator())
        self.no_iterations_input.setMaxLength(4)
        self.no_iterations_input.setPlaceholderText('No. iterations')
        self.no_iterations_input.setAlignment(Qt.AlignCenter)
        self.no_iterations_input.resize(180, 24)
        self.no_iterations_input.setFocus()
        self.no_iterations_input.move(670, 100)

        # No detectors label
        scan_angle_label = QLabel(self)
        scan_angle_label.setText('Scan angle in degrees')
        scan_angle_label.move(670, 130)

        # No iterations input
        self.scan_angle_input.setValidator(QIntValidator())
        self.scan_angle_input.setMaxLength(3)
        self.scan_angle_input.setPlaceholderText('Scan angle')
        self.scan_angle_input.setAlignment(Qt.AlignCenter)
        self.scan_angle_input.resize(180, 24)
        self.scan_angle_input.setFocus()
        self.scan_angle_input.move(670, 150)

        # Disable filter checkbox
        self.checkbox_filter.setChecked(False)
        self.checkbox_filter.move(670, 180)

        # MSE checkbox
        self.mse_checkbox.setChecked(False)
        self.mse_checkbox.move(670, 200)

        # Button read file
        self.button_read_file.resize(190, 35)
        self.button_read_file.move(665, 220)

        # Button run code
        self.button_run_code.setDisabled(True)
        self.button_run_code.resize(190, 35)
        self.button_run_code.move(665, 250)

        # Button again
        self.button_run_again.setDisabled(True)
        self.button_run_again.resize(190, 35)
        self.button_run_again.move(665, 280)

        # Input dicom date
        self.dicom_date_input.resize(180, 24)
        self.dicom_date_input.move(670, 320)
        self.dicom_date_input.setAlignment(Qt.AlignCenter)

        # Input patient ID
        self.patient_id_input.resize(180, 24)
        self.patient_id_input.move(670, 350)
        self.patient_id_input.setMaxLength(255)
        self.patient_id_input.setPlaceholderText('Patient ID')
        self.patient_id_input.setFocus()
        self.patient_id_input.setAlignment(Qt.AlignCenter)

        # Input patient name
        self.patient_name_input.resize(180, 24)
        self.patient_name_input.move(670, 380)
        self.patient_name_input.setMaxLength(255)
        self.patient_name_input.setPlaceholderText('Patient name')
        self.patient_name_input.setFocus()
        self.patient_name_input.setAlignment(Qt.AlignCenter)

        # Label selected iteration
        label_selected_iteration = QLabel(self)
        label_selected_iteration.setText('Select iteration to view')
        label_selected_iteration.move(670, 430)

        # Selected iteration input
        self.selected_iteration_input.setValidator(QIntValidator())
        self.selected_iteration_input.setMaxLength(4)
        self.selected_iteration_input.setPlaceholderText('Selected iteration number')
        self.selected_iteration_input.setAlignment(Qt.AlignCenter)
        self.selected_iteration_input.resize(180, 24)
        self.selected_iteration_input.setFocus()
        self.selected_iteration_input.move(670, 450)

        # Button show selected iteration
        self.button_show_selected.setDisabled(True)
        self.button_show_selected.resize(190, 35)
        self.button_show_selected.move(665, 480)

        # Button show MSE
        self.button_show_mse.setDisabled(True)
        self.button_show_mse.resize(190, 35)
        self.button_show_mse.move(665, 510)

        # Checkbox save as dicom
        self.save_dicom_checkbox.setChecked(False)
        self.save_dicom_checkbox.move(670, 600)

        # Show UI
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def filter_change(self):
        self.filter_enable = self.checkbox_filter.isChecked()

    def mse_change(self):
        self.mse_enable = self.mse_checkbox.isChecked()

    def save_dicom_change(self):
        self.save_as_dicom = self.save_dicom_checkbox.isChecked()

    def select_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Images (*.png *.dcm *.jpg)", options=options)
        if file_name:
            self.selected_file = file_name.lower()
            self.button_read_file.setText('Selected file')
            self.button_run_code.setDisabled(False)

    def validate_detectors(self):
        self.no_detectors = string_to_int(self.input_detectors.text())
        if self.no_detectors > 1:
            return True
        else:
            return False

    def validate_iterations(self):
        self.iterations = string_to_int(self.no_iterations_input.text())
        if self.iterations > 0:
            return True
        else:
            return False

    def validate_selected_iterations(self):
        self.selected_iteration = string_to_int(self.selected_iteration_input.text())
        if 0 < self.selected_iteration <= self.iterations:
            return True
        else:
            return False

    def validate_scan_angle(self):
        self.scan_angle = string_to_int(self.scan_angle_input.text())
        if 0 < self.scan_angle < 360:
            return True
        else:
            return False

    def show_mse_graph(self):
        # Enabled MSE calculations - toggle image
        if self.mse_enable:
            status = self.already_show_mse
            self.already_show_mse = not self.already_show_mse
        else:
            status = True

        self.mse_graph.setVisible(not status)
        self.mse_graph.setPixmap(QPixmap('results/mse_graph.jpg').scaled(640, 480, Qt.KeepAspectRatio))
        self.label_original_picture.setVisible(status)
        self.original_picture_basic.setVisible(status)
        self.label_sinogram_picture.setVisible(status)
        self.sinogram_picture_basic.setVisible(status)
        self.selected_picture_basic.setVisible(status)
        self.label_reverse_picture.setVisible(status)
        self.reverse_picture_basic.setVisible(status)
        self.label_selected_picture.setVisible(status)

    def show_selected_image(self, original):
        if original:
            self.selected_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))
        else:
            number = self.selected_iteration - 1
            self.selected_picture_basic.setPixmap(QPixmap('results/cut_' + str(number) + '.jpg').scaled(300, 300, Qt.KeepAspectRatio))

    def show_selected_file(self, original, extra_data=None):
        if original:
            self.original_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))
        else:
            if extra_data:
                self.original_picture_basic.setPixmap(QPixmap(extra_data).scaled(300, 300, Qt.KeepAspectRatio))
            else:
                self.original_picture_basic.setPixmap(QPixmap(self.selected_file).scaled(300, 300, Qt.KeepAspectRatio))

    def show_sinogram(self, original):
        if original:
            self.sinogram_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))
        else:
            self.sinogram_picture_basic.setPixmap(QPixmap('results/sinogram.jpg').scaled(300, 300, Qt.KeepAspectRatio))

    def show_reverse_image(self, original):
        if original:
            self.reverse_picture_basic.setPixmap(QPixmap('Files/image_not_available.jpg').scaled(300, 300, Qt.KeepAspectRatio))
        else:
            number = self.iterations - 1
            self.reverse_picture_basic.setPixmap(QPixmap('results/cut_' + str(number) + '.jpg').scaled(300, 300, Qt.KeepAspectRatio))

    def show_cut_file(self):
        if self.validate_selected_iterations():
            self.show_selected_image(False)
        else:
            print('Errors with parameters')

    def run_again(self):
        # Block button
        self.button_run_again.setDisabled(True)

        # Show again base pictures
        self.show_sinogram(True)
        self.show_selected_file(True)
        self.show_selected_image(True)
        self.show_reverse_image(True)

        # Read file button to default settings
        self.button_read_file.setText('Select file')
        self.button_read_file.setDisabled(False)

        # Unblock checkbox
        self.mse_checkbox.setDisabled(False)
        self.checkbox_filter.setDisabled(False)
        self.save_dicom_checkbox.setDisabled(False)

        # Block iterations options
        self.button_show_mse.setDisabled(True)
        self.button_show_selected.setDisabled(True)

    def run_code(self):
        if self.validate_detectors() and self.validate_iterations() and self.validate_scan_angle():
            if self.debug:
                print('Selected file => ', self.selected_file)
                print('Mask percent => ', self.no_detectors)
                print('No detectors => ', self.no_detectors)
                print('No iterations => ', self.iterations)
                print('Scan angle => ', self.scan_angle)
                print('Enabled filter => ', self.filter_enable)
                print('Enabled MSE => ', self.mse_enable)

            # Start time
            start_time = timer()

            # Block actions
            self.button_run_code.setDisabled(True)
            self.button_run_code.setText('In progress ...')
            self.button_read_file.setDisabled(True)
            self.mse_checkbox.setDisabled(True)
            self.checkbox_filter.setDisabled(True)
            self.save_dicom_checkbox.setDisabled(True)

            if self.debug:
                print('Settings [' + str(timer() - start_time) + ' s]')

            # Clear results
            ensure_exists_results_dir()
            clear_results()

            if self.debug:
                print('Clear & Check [' + str(timer() - start_time) + ' s]')

            original_file_data = None
            # Make program
            if self.selected_file.endswith('.dcm'):
                # DICOM read
                ds, file = read_dicom('result_dicom.dcm')
                height = ds.Columns
                width = ds.Rows

                original_file_data = QImage(file.tostring(), file.shape[0], file.shape[1], file.shape[0], QImage.Format_Mono)

                if self.debug:
                    debug_dicom(ds)
            else:
                # JPG
                file, height, width = read_file(self.selected_file)

            # Prepare
            picture, radius = prepare_circle(file, height, width)

            if self.debug:
                print('Read & circle [' + str(timer() - start_time) + ' s]')

            # Show original picture
            self.show_selected_file(False, extra_data=original_file_data)

            # Make base sinogram
            sinogram = make_sinogram(picture, radius, self.iterations, self.scan_angle, self.no_detectors)

            if self.debug:
                print('Sinogram [' + str(timer() - start_time) + ' s]')

            # Write sinogram to file
            write_file("sinogram", sinogram)

            # Show sinogram in program
            self.show_sinogram(False)

            # If filtered - apply filter
            if self.filter_enable:
                mask_size = int(self.no_detectors * self.mask_percent) + 1
                mask = prepare_ramlak_mask(mask_size)
                sinogram = make_convolve(sinogram, mask)

                if self.debug:
                    print('Filter [' + str(timer() - start_time) + ' s]')

            # Make reverse images and calculate MSE if enabled
            picture_from_reverse_sinogram, mse_errors = reverse_sinogram(self.mse_enable, file, picture, sinogram, radius, self.iterations, self.scan_angle, self.no_detectors, height, width)

            if self.debug:
                print('Reverse [' + str(timer() - start_time) + ' s]')

            # Show final
            self.show_reverse_image(False)

            # Operations on MSE
            if self.mse_enable:
                self.button_show_mse.setDisabled(False)
                max_mse = calculate_max_mse(file)
                prepare_mse_graph(max_mse, mse_errors)

                if self.debug:
                    print('MSE [' + str(timer() - start_time) + ' s]')

            # If selected save to DICOM
            if self.save_as_dicom:
                date = str(self.dicom_date_input.date().toPyDate())
                patient_name = self.patient_name_input.text()
                patient_id = self.patient_id_input.text()

                picture = cut_original_size(picture_from_reverse_sinogram, height, width)
                save_dicom(picture, 'result_dicom.dcm', date, patient_name, patient_id)

                if self.debug:
                    print('DICOM save [' + str(timer() - start_time) + ' s]')

            # End
            self.button_show_selected.setDisabled(False)
            self.button_run_again.setDisabled(False)
            self.button_run_code.setText('Run tomography')
        else:
            print('Errors with parameters')


def string_to_int(text):
    if text == '':
        return 0
    else:
        return int(text)


def ensure_exists_results_dir():
    if not os.path.exists('results'):
        os.makedirs('results')


def clear_results():
    file_list = [f for f in os.listdir('results')]
    for f in file_list:
        os.remove(os.path.join('results', f))


def debug_dicom(dataset):
    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name .....:", display_name)
    print("Patient ID .........:", dataset.PatientID)
    print("Content Date .......:", dataset.ContentDate)


def read_dicom(file_name):
    # Read all details
    ds = pydicom.dcmread(file_name)

    # Parse strings to int16
    parsed = np.fromstring(ds.PixelData, dtype=np.uint16)

    # Calculate max value and fetch shape
    max_value = parsed.max()
    columns = ds.Columns
    rows = ds.Rows

    # Prepare pixel array content
    result = np.empty((columns, rows), dtype=np.uint8)

    # Variables for loop
    counter = 0
    row = 0

    # Loop to prepare correct int8 values in array
    for ind, value in enumerate(parsed):
        result[counter][row] = int((value * 256) / max_value)
        counter += 1
        if counter == columns:
            counter = 0
            row += 1

    # Return
    return ds, result


def save_dicom(pixel_array, file_name, date, patient_name, patient_id):
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"

    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = str(patient_name)
    ds.PatientID = str(patient_id)
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.ContentDate = str(date.replace('-', ''))
    ds.ContentTime = str(int(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()))

    ds.save_as(file_name)


def main():
    app = QApplication(sys.argv)
    ex = Tomography()
    sys.exit(app.exec_())


if __name__ == "__main__": main()