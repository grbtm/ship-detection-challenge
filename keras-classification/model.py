import keras
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.applications.densenet import DenseNet121
from sklearn.metrics import average_precision_score
import tensorflow as tf

import keras.backend as K


def resnet(input_shape):
    vanilla_resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape,
                              pooling='avg')

    x = vanilla_resnet.outputs[0]
    x = Dense(1, activation='sigmoid', name='prob')(x)

    model = Model(inputs=vanilla_resnet.inputs, outputs=[x])

    return model


def densenet(input_shape):
    vanilla_densenet = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
    x = vanilla_densenet.outputs[0]
    x = Dense(1, activation='sigmoid', name='prob')(x)

    model = Model(inputs=vanilla_densenet.inputs, outputs=[x])

    return model
