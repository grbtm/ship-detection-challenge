

from sliding_window import *
from utils import *
import cv2



image_name = "Pipstrel-Virus_Bodensee_2018-02-13_15-41-05.jpg"
xmin = 2501    
ymin = 1537    
xmax = 2528    
ymax = 1566

bb_list = [[xmin,ymin],
           [xmax,ymax]]



img = cv2.imread(image_name)
aug_image,bb_image = deep_data_aug(img,bb_list)
slide_class = SlidingWindow()
sub_figures,labels_boat_or_no_boat = slide_class.return_sub_figures_with_labels(aug_image,bb_image)

