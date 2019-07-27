import os

import pandas as pd
from keras.utils import multi_gpu_model

from model import resnet
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint

if __name__ == '__main__':
    train_img_dir = '../../data/searchwing/airbus_ship_data/train_v2'
    test_img_dir = '../../data/searchwing/Hackathon/SingleFrame_ObjectProposalClassification/test'
    train_csv = '../../data/searchwing/airbus_ship_data/train_zero_to_one_ships.csv'
    test_csv = '../../data/searchwing/Hackathon/SingleFrame_ObjectProposalClassification/test/pipistrel_image_cls.csv'
    log_dir = '../logs'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    image_size = (224, 224)
    input_shape = image_size + (3,)
    gpus = 4
    batch_size = 16 * gpus

    model = resnet(input_shape)

    # todo freeze features


    multi_gpu_model(model, gpus=gpus)

    model.compile(optimizer=Adam(lr=3e-4), loss=binary_crossentropy, metrics=['accuracy'])

    image_generator = ImageDataGenerator()

    tensorboard = TensorBoard(log_dir, batch_size=batch_size)

    # chkp_saver = ModelCheckpoint(log_dir + 'weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='loss')

    model.fit_generator(image_generator.flow_from_dataframe(train_df, train_img_dir, x_col='ImageId', y_col='label',
                                                            class_mode='binary', batch_size=batch_size,
                                                            target_size=image_size, save_to_dir=log_dir),
                        epochs=1,
                        steps_per_epoch=len(train_df) // batch_size // gpus,
                        validation_steps=len(test_df) // batch_size // gpus,
                        callbacks=[tensorboard],
                        validation_data=image_generator.flow_from_dataframe(test_df, test_img_dir, x_col='ImageId',
                                                                            y_col='class',
                                                                            class_mode='sparse', batch_size=batch_size,
                                                                            target_size=image_size))

    model.save_weights(os.path.join(log_dir, 'trained.h5'))
