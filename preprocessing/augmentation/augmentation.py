#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Augmentation Function
def deep_data_aug(image,bb_list):
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
   
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=bb_list[0][0], 
                    y1=bb_list[0][1], 
                    x2=bb_list[1][0], 
                    y2=bb_list[1][1]),
    ], shape=image.shape)
    
    
    seq = iaa.Sometimes(0.5,iaa.SomeOf((0, None),[
        iaa.Fliplr(0.3),
        iaa.Flipud(0.2),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
        ], random_order=True))
    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    formated_bbs_aug = [[bbs_aug.bounding_boxes[0].x1,
                        bbs_aug.bounding_boxes[0].y1,
                        bbs_aug.bounding_boxes[0].x2,
                        bbs_aug.bounding_boxes[0].y2]]
    return image_aug,formated_bbs_aug

data_dir = "./"
imgs_dir = "train_imgs/"
output_dir = "aug_imgs/"
labels_train = pd.read_csv(data_dir + "labelsTrain.csv")
counter = 0
aug_labels = []
for _, row in labels_train.iloc[:20,:].iterrows():
    image_name = row[0]
    xmin = row[2]   
    ymin = row[3]    
    xmax = row[4] 
    ymax = row[5]

    bb_list = [[xmin,ymin],
            [xmax,ymax]]



    img = cv2.imread(data_dir + imgs_dir + image_name)
    aug_image,bb_image = deep_data_aug(img,bb_list)
    filename = "augmented_bodensee_" + str(counter) + ".jpg"
    aug_labels.append((filename, row[1],xmin, ymin, xmax, ymax))
    plt.imsave(data_dir + output_dir + filename,np.array(aug_image))
    counter += 1

aug_labels = pd.DataFrame(aug_labels, columns=['filename','class', 'xmin', 'ymin', 'xmax', 'ymax'])
aug_labels.to_csv(data_dir + "augmented_labels.csv", index=False)


# In[ ]:




