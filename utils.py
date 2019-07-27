def deep_data_aug(image,xmin,ymin,xmax,ymax):
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    
    
    # =============================================================================
    # image = ia.quokka(size=(256, 256))
    # bbs = BoundingBoxesOnImage([
    #     BoundingBox(x1=65, y1=100, x2=200, y2=150),
    #     BoundingBox(x1=150, y1=80, x2=200, y2=130)
    # ], shape=image.shape)
    # 
    # =============================================================================
    
    #image = 
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax),
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
    return image_aug,bbs_aug
