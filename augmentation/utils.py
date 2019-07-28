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
