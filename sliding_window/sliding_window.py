import cv2
import math
import numpy as np


class SlidingWindow():

    def return_sub_figures(img):
        """ Convert an image to a list of subfigures

        :param image: [numpy.ndarray] image
        :return: [numpy.ndarray] containing subfigures
        """

        image_width = float(img.shape[1])
        image_height = float(img.shape[0])

        y_size = 200
        x_size = 200
        overlap = 0.5 * x_size
        x_number_windows = round(image_width / overlap)

        y_number_windows = round(image_height / overlap)

        overlap_x = math.floor((image_width - x_size) / x_number_windows)
        overlap_y = math.floor((image_height - y_size) / y_number_windows)

        image_list = list()

        for j in np.arange(0, y_number_windows):
            for i in np.arange(0, x_number_windows):
                x = i * overlap_x
                y = j * overlap_y
                sub_figure = img[y:y + y_size, x:x + x_size]
                image_list.append(sub_figure)

        return np.array(image_list)
