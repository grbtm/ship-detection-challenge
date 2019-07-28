
import math
import numpy as np


class SlidingWindow():
    @staticmethod
    def return_sub_figures(self,img, subfig_width=200, subfig_height=200, overlap_percentage=0.5):
        """ Convert an image to a list of subfigures
    
        :param img: [numpy.ndarray] image
        :param subfig_width: [int] representing width of subfigures
        :param subfig_height: [int] representing height of subfigures
        :param overlap_percentage: [float] overlap of sliding windows in percentage
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
    
    def return_sub_figures_with_labels(self,img, bounding_coords, subfig_width=200, subfig_height=200,
                                       overlap_percentage=0.5, min_bbox_overlap_percentage=0.25):
        """ Convert an image to a list of subfigures
    
        :param img: [numpy.ndarray] image
        :param bounding_coords: [Lst[Lst[int]]] List of bounding boxes, each box contains the pixels of the bounding
                                box lines from the upper left corner of the picture (xmin, ymin, xmax, ymax)
        :param subfig_width: [int] width of subfigures
        :param subfig_height: [int] height of subfigures
        :param overlap_percentage: [float] overlap of sliding windows in percentage
        :param min_bbox_overlap_percentage: [float] minimum overlap of the sliding window and the bounding box in
                                            percentage, to mark the picture as containing any boat
        :return: [numpy.ndarray] containing subfigures,
                [Lst[int]] of same length as array of subfigures, including zeros for no boat in subfigure and
                ones for any boat in subfigure
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
    
                self._validation_of_intersection(bounding_coords, height, index_height, index_width,
                                            min_bbox_overlap_percentage, number_subfigs_along_width,
                                            sliding_x2, sliding_y2, validation_list, width)
    
                image_list.append(sub_figure)
    
        return np.array(image_list), validation_list
    
    def _validation_of_intersection(self,bounding_coords, sliding_y1, index_y, index_x,
                                    min_bbox_overlap_percentage,
                                    x_number_subfigs,
                                    sliding_x2, sliding_y2,
                                    validation_list, sliding_x1):
        for bbox in bounding_coords:
    
            if bbox[0] < sliding_x2:
                if bbox[2] > sliding_x1:
                    if bbox[1] < sliding_y2:
                        if bbox[3] > sliding_y1:
                            x_left = max(sliding_x1, bbox[0])
                            y_top = max(sliding_y1, bbox[1])
                            x_right = min(sliding_x2, bbox[2])
                            y_bottom = min(sliding_y2, bbox[3])
    
                            # The intersection of two axis-aligned bounding boxes is always an
                            # axis-aligned bounding box
                            intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
                            if intersection_area > min_bbox_overlap_percentage:
                                validation_list[index_y * x_number_subfigs + index_x] = 1
                                continue

