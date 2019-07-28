This set of functions is made for preprocessing of high resolution photographs using the Sliding Window approach.
These functions are designed to be used for training and production of deep learning neural networks.

The image_and_bounding_boxes_generator function provides images and the belonging bounding boxes of detected boats,
including undefined objects.
Hence, the return_sub_figures_with_labels function generates multiple subfigures (the sliding windows!) of an input
photograph. These subfigures are labeled as containing any boat (=1) or no boat (=0).

The return_sub_figures function generates subfigures in the same way as the return_sub_figures_with_labels function, but
without labels.