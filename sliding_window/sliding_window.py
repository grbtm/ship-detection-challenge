# This bundle of functions is free to use for any non-commercial use. It was developed by Sarah Schroeder and Tomislav
# Grbesic.

import os
import cv2
import math
import numpy as np
import pandas as pd


def _get_bounding_boxes(df, img_name):
    df_of_image = df.loc[df.filename == img_name]
    df_of_boats = df_of_image[(df_of_image["class"] == "boat") | (df_of_image["class"] == "undefObj")]
    return df_of_boats[["xmin", "ymin", "xmax", "ymax"]].values.tolist()


def get_bounding_boxes_of_all_images(csv_path):
    """ Return dictionary that maps each image filename to it's list of bounding boxes

    :param csv_path: [Str] path to csv file including the file name
    :return dict_of_bounding_boxes: [Dict[Str->[Lst[Lst[int]]]]] that maps image filenames to respective bounding boxes
    """
    df = pd.read_csv(csv_path)
    all_images = df["filename"].unique()
    dict_of_bounding_boxes = dict()
    for image_name in all_images:
        bbox_list = _get_bounding_boxes(df, image_name)
        dict_of_bounding_boxes[image_name] = bbox_list
    return dict_of_bounding_boxes


def image_and_bounding_boxes_generator(csv_path, images_path):
    """ Generator returning one cv2 image and its respective list of boundary boxes

    :param csv_path: [Str] path to csv file including the file name
    :param images_path: [Str] path to directory containing all images
    :yield image: [numpy.ndarray] of the image,
            bbox_list: [Lst[Lst[int]]] list of bounding boxes, each box [Lst[int]] contains the pixels of the bounding
                            box lines from the upper left corner of the image [xmin, ymin, xmax, ymax]
    """
    bounding_boxes_of_images_dict = get_bounding_boxes_of_all_images(csv_path)
    for image_name in bounding_boxes_of_images_dict:

        image = cv2.imread(os.path.join(images_path, image_name))
        if image is None:
            print("Filename: {name} not found in directory: {path}!".format(name=image_name, path=images_path))
            continue
        bbox_list = bounding_boxes_of_images_dict[image_name]
        yield image, bbox_list

#TODO write wrapper function to write the subfigers to a dir with a csv containing the labels

def return_sub_figures(img, subfig_width=200, subfig_height=200, overlap_percentage=0.5):
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


def return_sub_figures_with_labels(img, bounding_coords, subfig_width=200, subfig_height=200,
                                   overlap_percentage=0.5, min_bbox_overlap_percentage=0.25):
    """ Convert an image to a list of subfigures

    :param img: [numpy.ndarray] image
    :param bounding_coords: [Lst[Lst[int]]] List of bounding boxes, each box [Lst[int]] contains the pixels of the
                            bounding box lines from the upper left corner of the picture [xmin, ymin, xmax, ymax]
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

            _validation_of_intersection(bounding_coords, height, index_height, index_width,
                                        min_bbox_overlap_percentage, number_subfigs_along_width,
                                        sliding_x2, sliding_y2, validation_list, width)

            image_list.append(sub_figure)

    return np.array(image_list), validation_list


def _validation_of_intersection(bounding_coords, sliding_y1, index_y, index_x,
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
                            break
