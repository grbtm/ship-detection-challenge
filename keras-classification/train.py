import os

import pandas as pd
from keras.utils import multi_gpu_model

from model import resnet, densenet
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras_metrics as km

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', type=str, required=True)
    parser.add_argument('--test_img_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True, default=3e-4)
    parser.add_argument('--mini_batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--densenet', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # train_img_dir = '../../data/searchwing/airbus_ship_data/train_v2'
    # test_img_dir = '../../data/searchwing/Hackathon/RAW_DATA'
    # train_csv = '../../data/searchwing/airbus_ship_data/train_zero_to_one_ships.csv'
    # test_csv = '../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv'
    # log_dir = '../logs'

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    image_size = (224, 224)
    input_shape = image_size + (3,)
    gpus = 4

    batch_size = args.mini_batch_size * gpus

    model = resnet(input_shape)
    # freeze features
    if args.freeze:
        for layer in model.layers[:-2]:
            layer.trainable = False
    if args.densenet:
        model = densenet(input_shape)


    model = multi_gpu_model(model, gpus=gpus)

    model.compile(optimizer=Adam(lr=args.lr), loss=binary_crossentropy,
                  metrics=['accuracy', km.binary_precision(), km.binary_recall()])

    image_generator = ImageDataGenerator()

    tensorboard = TensorBoard(args.log_dir, batch_size=batch_size)

    # chkp_saver = ModelCheckpoint(log_dir + 'weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='loss')

    model.fit_generator(
        image_generator.flow_from_dataframe(train_df, args.train_img_dir, x_col='ImageId', y_col='class',
                                            class_mode='binary', batch_size=batch_size,
                                            target_size=image_size),
        epochs=args.epochs,
        steps_per_epoch=len(train_df) // (batch_size),
        validation_steps=len(test_df) // (batch_size),
        callbacks=[tensorboard],
        validation_data=image_generator.flow_from_dataframe(test_df, args.test_img_dir, x_col='ImageId',
                                                            y_col='class',
                                                            class_mode='binary', batch_size=batch_size,
                                                            target_size=image_size))

    model.save_weights(os.path.join(args.log_dir, 'trained.h5'))
