import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


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
    diagonal = int(math.sqrt((height ** 2) + (width ** 2))) + 5
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

    # TODO add way to calculate positions when passed only 1 detector - now raise error divide by zero.
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
    return detectors


def main():
    # TODO move parameters as named program's parameters
    detectors_counter = 2
    scan_angle = 180
    iterations = 10

    filename = "Files/Kolo.jpg"
    file, height, width = read_file(filename)
    picture, radius = prepare_circle(file, height, width)
    # write_file("example", file)
    # write_file("example_test", test)

    # print(calculate_iterations_angle(iterations))
    # print(calculate_angle_between_detectors(detectors_counter, scan_angle))

    alpha = 270
    alpha_in_radians = np.radians(alpha)
    emitter = emitter_position(radius, alpha_in_radians, radius)

    scan_angle_in_radius = np.radians(scan_angle)
    print(detectors_position(radius, alpha_in_radians, scan_angle_in_radius, radius, detectors_counter))


if __name__ == "__main__":
    main()

    # Prepared in code
    # * Read file [DONE]
    # * Move data to circle with extra size [DONE]
    # * Calculation angle per step to have 360 degrees [DONE]
    # * Calculation detector's angle (between two detectors) [DONE]
    # * Function to calculation positions on image our detectors and emitter [DONE]

    # TODO
    # * Bresenham line interpolation
    # * Calculation sum of values on line emitter --> detector
    # * Make sinogram
    # * Filtered sinogram
    # * Write sinogram to file
    # * Calculation MSE in percent (current / max_errors_possible)
    # * Reverse Radon transformation - from sinogram make picture
    # * Save result to file with calculated MSE on it.
    # * Display mode - extra parameter to disable calculations and show only results
    # * GUI
    # * Convolve - own, not from library files
