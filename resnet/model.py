from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense


def resnet(input_shape):
    vanilla_resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape,
                              pooling='avg')

    x = vanilla_resnet.outputs[0]
    x = Dense(1, activation='sigmoid', name='prob')(x)

    model = Model(inputs=vanilla_resnet.inputs, outputs=[x])

    return model
