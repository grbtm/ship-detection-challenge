#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import average_precision_score
import argparse

if __name__ == "__main__":
     # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Calc Average Precision for image classification")
    parser.add_argument("-g",
                        "--gtfile",
                        help="Filepath of ground truth",
                        type=str)
    parser.add_argument("-d",
                        "--detfile",
                        help="Filepath of detection file", type=str)
    args = parser.parse_args()
    
    gt_df = pd.read_csv(args.gtfile)
    classT = {'nature': 0.0,'boat': 1.0} 
    gt_df['class'] = [classT[item] for item in gt_df['class']] 
    
    det_df = pd.read_csv(args.detfile)
        
    df = pd.merge(left=gt_df, right=det_df, how='left', left_on='filename', right_on = 'filename')
    
    AP=average_precision_score(df['class'], df['confidence'], average="macro", sample_weight=None)
    print("Average precision:",str(round(AP, 2)))
