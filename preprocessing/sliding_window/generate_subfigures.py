import argparse
import preprocessing.sliding_window.functions_and_generators as sw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', type=str, required=True)
    parser.add_argument('--test_img_dir', type=str, required=True)
    parser.add_argument('--train_bbox_csv', type=str, required=True)
    parser.add_argument('--test_bbox_csv', type=str, required=True)
    parser.add_argument('--subfig_width', type=int, required=False)
    parser.add_argument('--subfig_height', type=int, required=False)
    parser.add_argument('--overlap_percentage', type=float, required=False)
    parser.add_argument('--min_bbox_overlap_percentage', type=float, required=False)
    parser.add_argument('--train_output_csv', type=str, required=True)
    parser.add_argument('--test_output_csv', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    return parser.parse_args()


# TODO where is labelsTest_complete.csv? we only have labelsTrain_complete.csv
if __name__ == '__main__':
    args = parse_args()
    args.train_csv

    generator = sw.image_and_bounding_boxes_generator(csv_path, images_path)
    for image_name, image, bbox_list in generator:
        # (np.array(image_list), validation_list) = return_sub_figures_with_labels(img, bounding_coords, subfig_width=200,
        #                                                                         subfig_height=200, overlap_percentage=0.5,
        #                                                                         min_bbox_overlap_percentage=0.25)
        subfigures_array, labels_array =
