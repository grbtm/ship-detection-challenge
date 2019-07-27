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

    def return_sub_figures_with_labels(img, bounding_coords, subfig_width=200, subfig_height=200,
                                       overlap_percentage=0.5, min_bbox_overlap_percentage=0.25):
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
        nr_subfigs = number_subfigs_along_width * number_subfigs_along_height
        validation_list = [0] * nr_subfigs

        for index_height in np.arange(0, number_subfigs_along_height):
            for index_width in np.arange(0, number_subfigs_along_width):
                width = index_width * overlap_along_width
                height = index_height * overlap_along_height
                sliding_y2 = height + subfig_height
                sliding_x2 = width + subfig_width

                sub_figure = img[height:sliding_y2, width:sliding_x2]

                for bbox in bounding_coords:

                    if bbox[0] < width:
                        if bbox[2] > sliding_x2:

                            x_left = max(width, bbox[0])
                            y_top = max(height, bbox[1])
                            x_right = min(sliding_x2, bbox[2])
                            y_bottom = min(sliding_y2, bbox[3])

                            # The intersection of two axis-aligned bounding boxes is always an
                            # axis-aligned bounding box
                            intersection_area = (x_right - x_left) * (y_bottom - y_top)

                            if intersection_area > min_bbox_overlap_percentage:
                                validation_list[index_height * number_subfigs_along_width + index_width + 1] = 1
                                continue

                image_list.append(sub_figure)

        return np.array(image_list), validation_list
