import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham


# TODO debug values
debug = True
images = False


def read_file(filename):
    # Read file in gray scale
    file = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Check size of original image
    height, width = file.shape
    # Return detail
    return file, height, width


def prepare_circle(file, height, width):
    # Calculation diagonal
    diagonal = int(math.sqrt(math.pow(height, 2) + math.pow(width, 2))) + 10
    # Append empty values outside the image
    image = np.zeros((diagonal, diagonal), dtype=np.uint8)

    # Padding in view
    padding_y = int((diagonal - height) / 2)
    padding_x = int((diagonal - width) / 2)

    # Append original picture
    image[padding_y: padding_y + height, padding_x: padding_x + width] = file

    # Calculation radius
    radius = int(diagonal / 2)

    # Draw circle
    if debug:
        center = (int(diagonal / 2), int(diagonal / 2))
        cv2.circle(image, center, radius, 100)

    if images:
        plt.imshow(image)
        plt.show()

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

    # Loop to fetch values from picture
    for position in line:
        value += picture.item(position)

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


def reverse_sinogram(original, picture, sinogram, radius, no_iterations, scan_angle, no_detectors, height, width):
    # Prepare zeros array with the same size like picture
    result = np.zeros_like(picture, dtype=float)
    mse_errors = np.zeros(no_iterations)

    # Angle per iteration, scan angle in radius
    angle_per_iteration = calculate_iterations_angle(no_iterations)
    scan_angle_in_radius = np.radians(scan_angle)

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
                # Append values to image
                result[position_x, position_y] += sinogram[iteration, index]

        # Save iteration results
        write_file("reverse_" + str(iteration), result)
        picture = cut_original_size(result, height, width)
        write_file("cut_" + str(iteration), picture)

        # Calculate MSE for iteration
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


def main():
    # TODO move parameters as named program's parameters
    detectors_counter = 360
    scan_angle = 180
    iterations = 360
    mask_size = int(detectors_counter * 0.20) + 1
    filtered = True

    filename = "Files/Kwadraty2.jpg"
    file, height, width = read_file(filename)
    picture, radius = prepare_circle(file, height, width)

    sinogram = make_sinogram(picture, radius, iterations, scan_angle, detectors_counter)
    write_file("sinogram", sinogram)

    mask = prepare_ramlak_mask(mask_size)

    # If filtered - convolve sinogram and mask
    if filtered:
        sinogram = make_convolve(sinogram, mask)

    picture_from_reverse_sinogram, mse_errors = reverse_sinogram(file, picture, sinogram, radius, iterations, scan_angle, detectors_counter, height, width)
    write_file("reverse", picture_from_reverse_sinogram)
    max_mse = calculate_max_mse(file)
    prepare_mse_graph(max_mse, mse_errors)
    print(max_mse)
    print(mse_errors)


if __name__ == "__main__":
    main()

    # Prepared in code
    # * Read file [DONE]
    # * Move data to circle with extra size [DONE]
    # * Calculation angle per step to have 360 degrees [DONE]
    # * Calculation detector's angle (between two detectors) [DONE]
    # * Function to calculation positions on image our detectors and emitter [DONE]
    # * Bresenham line interpolation [DONE]
    # * Calculation sum of values on line emitter --> detector [DONE]
    # * Make sinogram [DONE]
    # * Normalize values in sinogram [DONE]
    # * Write sinogram to file [DONE]
    # * Reverse Radon transformation - from sinogram make picture [DONE]
    # * Return size of result to original from base picture [DONE]
    # * Calculate max MSE to proportions in formula [DONE]
    # * Calculation MSE in percent (current / max_errors_possible) [DONE]
    # * Save result to file with calculated MSE on it [DONE]
    # * Calculate mask - Ram-Lak filter [DONE]
    # * Convolve - own, not from library files [DONE]
    # * Filtered sinogram [DONE]
    # * Rewrite MSE to speed up function [DONE]
    # * Check is correct MSE implementation [DONE]

    # TODO
    # * Display mode - extra parameter to disable calculations and show only results
    # * GUI

