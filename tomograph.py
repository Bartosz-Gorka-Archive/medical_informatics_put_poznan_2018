import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    # Read file in gray scale
    file = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Check size of original image
    height, width = file.shape
    # Return detail
    return file, height, width


def draw_circle(file, height, width):
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
    center = (int(diagonal / 2), int(diagonal / 2))
    cv2.circle(image, center, radius, 100)

    # Show prepared data
    plt.imshow(image)
    plt.show()
    return image


def main():
    filename = "Files/Kolo.jpg"
    file, height, width = read_file(filename)
    draw_circle(file, height, width)


if __name__ == "__main__":
    main()

    # TODO-s list
    # 1. Write data to file
    # 2. Code (project) settings
