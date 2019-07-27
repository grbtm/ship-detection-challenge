import math
import numpy as np


class SlidingWindow():

    def return_sub_figures(img, subfig_width=200, subfig_height=200, overlap_percentage=0.5):
        """ Convert an image to a list of subfigures

        :param img: [numpy.ndarray] image
        :param subfig_width: [int] representing width of subfigures
        :param subfig_height: [int] representing height of subfigures
        :return: [numpy.ndarray] containing subfigures
        """

        image_width = float(img.shape[1])
        image_height = float(img.shape[0])

        overlap = overlap_percentage * subfig_width
        number_subfigs_along_width = round(image_width / overlap)
        number_subfigs_along_height = round(image_height / overlap)

        overlap_along_width = math.floor((image_width - subfig_width) / number_subfigs_along_width)
        overlap_along_height = math.floor((image_height - subfig_height) / number_subfigs_along_height)

        image_list = list()

        for index_height in np.arange(0, number_subfigs_along_height):
            for index_width in np.arange(0, number_subfigs_along_width):
                width = index_width * overlap_along_width
                height = index_height * overlap_along_height
                sub_figure = img[height:height + subfig_height, width:width + subfig_width]
                image_list.append(sub_figure)

        return np.array(image_list)
