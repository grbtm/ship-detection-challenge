# Classification with Keras

This subfolder contains the code of Doreen and Daniel.

We trained vanilla classification models on the Airbus dataset.

## Overview

*model* containing the code for the models

*train* the script to train the models

*test* used for evaluating on different datasets, calculates metrics  

*generate_csv_reflection* produces a csv file with class annotations for the images taken during the flight in the mediterranean. There is no image in the folder with a boat. The 'class' column will be filled with 'nature'. To use it with the keras ImageDataGenerator and flow_from_dataframe there has to be at least one row
with a 'boat' annotation. We added it by hand.

*pipistrel_to_df* create a csv file for the objectpProposal classification. The image files were in the Hackathon/SingleFrame_ObjectProposalClassification/test folder. Images were split into classes by folder containing them. To use the flow_from_dataframe we needed them in one folder and a csv file with a class annotation.

*MetricCallBack* was unused. Maybe useful to integrate metrics on multiple datasets into the training process.

Shell scripts were used to train and test.

## Best Result in dense_airbus_0-4

This was our best model. A DenseNet101.

The output of the model is a bit weired. We used the keras flow_from_dataframe function and it selected boat as negative and nature as positive. So if the prediction i 0 it means 100% confident in boat and 1 means 100% confident in nature.

Training:
- pretrained features on imagenet
- airbus 0-4 dataset
- lr: 3e-4
- 2 epochs

Performance on different datasets:

**Bodensee test**

AP: 0.999143

Accuracy: 0.991391

Precision: 0.9977375

Recall: 0.988789

Confusion Matrix:

 &nbsp;     | Pred: boat | Pred: nature
 ---        |---         |---
 Gt: boat   |    250     | 1
 Gt: nature |    5       | 441
 
 
**Mediterrenaean reflection set**

AP: 0.999502

Accuracy:  0.991404

Precision: 0.999422

Recall: 0.991972

Confusion Matrix:

 &nbsp;     | Pred: boat | Pred: nature
 ---        |---         |---
 Gt: boat   |    0       | 0
 Gt: nature |    14      | 1731

 **Bodensee object proposals test**

AP: 0.963029

Accuracy:  0.810834

Precision: 0.888095

Recall: 0.902492

Confusion Matrix:

 &nbsp;     | Pred: boat | Pred: nature
 ---        |---         |---
 Gt: boat   |    12      | 470
 Gt: nature |    403     | 3730
