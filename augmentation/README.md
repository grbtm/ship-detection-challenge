# Augmentation Examples

For all examples it is necessary to download the Pipstrel data set (https://cloud.hs-augsburg.de/index.php/s/mGHrPeW9WpKCYp5) and labels (https://cloud.hs-augsburg.de/index.php/s/YT8jEMieYM4EL5b) from the subfolder "SingleFrame_ObjectProposalClassification".

## Basic Augmenter

The basic augmenter is using the `imgaug` library and generates images using basic augmentation operations from the originals one by one and stores them in a target directory specified in `augmentation.py` and `augmentation.ipynb`.

## AutoAugment

The example for AutoAugment takes in an image and returns an image with a random policy applied to the image. See `basic_autoaugment_example.ipynb`
