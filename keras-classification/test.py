import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import multi_gpu_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score

from model import resnet, densenet
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import keras_metrics as km
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_img_dir', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True, help='folder with the weights')
    parser.add_argument('--log_file', type=str, required=True, help='file the results are written to')
    parser.add_argument('--densenet', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # test_img_dir = '../../data/searchwing/Hackathon/RAW_DATA'
    # test_csv = '../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv'
    classes = ['boat', 'nature']
    test_df = pd.read_csv(args.test_csv)

    threshold = 0.5

    image_size = (224, 224)
    input_shape = image_size + (3,)
    batch_size = 16

    model = resnet(input_shape)
    if args.densenet:
        model = densenet(input_shape)

    model = multi_gpu_model(model, gpus=4)

    model.load_weights(os.path.join(args.log_dir, 'trained.h5'))

    model.compile(optimizer=Adam(lr=3e-4), loss=binary_crossentropy,
                  metrics=['accuracy', km.binary_precision(), km.binary_recall()])

    image_generator = ImageDataGenerator()

    test_x = []
    test_y = []
    gen = image_generator.flow_from_dataframe(test_df, args.test_img_dir, x_col='ImageId',
                                              y_col='class',
                                              class_mode='binary', batch_size=1,
                                              target_size=image_size)
    for _ in range(len(test_df)):
        tup = gen.__next__()
        test_x.append(tup[0][0])
        test_y.append(tup[1][0])

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    predictions = []

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i, tup in enumerate(zip(test_x, test_y)):
        img, y = tup
        y_pred = model.predict(np.expand_dims(img, axis=0))
        y_pred = np.squeeze(y_pred)
        predictions.append(y_pred)
        # tp
        if y_pred < threshold and y == 0.0:
            tp += 1
        # fp
        if y_pred < threshold and y == 1.0:
            fp += 1
        # tn
        if y_pred > threshold and y == 1.0:
            tn += 1
        # fn
        if y_pred > threshold and y == 0.0:
            fn += 1

        if (y_pred > threshold and y == 0.0) or (y_pred < threshold and y == 1.0):
            print('found wrong %s' % classes[int(y)])

            image = Image.fromarray(img.astype('uint8'))
            image.save(os.path.join(args.log_dir, 'wrong_%d_%s.jpg' % (i, classes[int(y)])))
        else:
            image = Image.fromarray(img.astype('uint8'))
            image.save(os.path.join(args.log_dir, 'right_%d_%s.jpg' % (i, classes[int(y)])))

    AP = average_precision_score(test_y, predictions, average="macro", sample_weight=None)
    print('AP: %f' % AP)

    acc = model.evaluate(test_x, test_y)

    acc = [str(x) for x in acc]
    print('Evaluate Results: ', acc)

    print('Confusion matrix')
    print('TP: %d FP: %d TN: %d FN: %d' % (tp, fp, tn, fn))

    result = []
    result.append('AP: %f\n' % AP)
    result.append('Evaluate Results: %s\n' % ', '.join(acc))
    result.append('TP: %d FP: %d TN: %d FN: %d\n' % (tp, fp, tn, fn))

    with open(args.log_file, 'w+') as file:
        file.writelines(result)
