#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

ENVIRONMENT = os.environ.get('ENVIRONMENT')

get_ipython().system('pip install -U tensorflow_hub')
if ENVIRONMENT == 'DGX':
    get_ipython().system('pip install -U tensorflow_hub==0.5.0')
    get_ipython().system('pip install tensorflow-gpu==1.14.0')
else:
    get_ipython().system('pip install -U tensorflow_hub')
    get_ipython().system('pip install tf-nightly-gpu')

# In[2]:
from __future__ import absolute_import, division, print_function, unicode_literals

# In[3]:


import tensorflow as tf

tf.enable_eager_execution()
import tensorflow_hub as hub
from tensorflow.keras import layers

# In[4]:
if ENVIRONMENT == 'DGX':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    session = tf.Session(config=config)
    import tensorflow.keras.backend as K

    K.set_session(session=session)

# In[5]:
import matplotlib.pylab as plt
import numpy as np
import time
import os

# In[6]:

if ENVIRONMENT == 'DGX':
    os.environ['DATA_PATH'] = '/notebooks/data/datasets/pipistrel/Hackathon/SingleFrame_ObjectProposalClassification'
else:
    os.environ['DATA_PATH'] = '/home/badc0ded/notebooks/data'

# In[8]:


DATA_PATH = os.environ.get('DATA_PATH')

IMAGE_SHAPE = (224, 224)


# In[9]:
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


def read_data(data_root):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
    return image_data


def plot_predicted_batch(image_batch, label_batch, predicted_batch):
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]
    label_id = np.argmax(label_batch, axis=-1)
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        color = "green" if predicted_id[n] == label_id[n] else "red"
        plt.title(predicted_label_batch[n].title(), color=color)
        plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")


def eval_model_on_data(model, data_root_short_name):
    data_root = os.path.join(DATA_PATH, data_root_short_name)
    image_data = read_data(data_root)
    loss, acc = model.evaluate(image_data)
    print('Accuracy: {}'.format(acc))
    return loss, acc, image_data


    # In[10]:

def main():
    train_data_root = os.path.join(DATA_PATH, "train")
    train_image_data = read_data(train_data_root)
    train_image_batch, train_label_batch = train_image_data[0]

    # In[11]:

    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"  # @param {type:"string"}
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=IMAGE_SHAPE + (3,))
    feature_extractor_layer(train_image_batch)
    feature_extractor_layer.trainable = False

    # In[12]:

    model = tf.keras.Sequential([
        # layers.Dropout(0.2, input_shape=(224,224,3)),
        feature_extractor_layer,
        # layers.Dense((train_image_data.num_classes +  1280) / 2, activation = 'relu'),
        layers.Dense(train_image_data.num_classes, activation='softmax')
    ])

    model.summary()

    # In[ ]:

    predictions = model(train_image_batch)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['acc'])

    steps_per_epoch = np.ceil(train_image_data.samples / train_image_data.batch_size)

    batch_stats_callback = CollectBatchStats()

    history = model.fit(train_image_data, epochs=2,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[batch_stats_callback])

    # In[ ]:

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats_callback.batch_losses)

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats_callback.batch_acc)

    # In[ ]:

    class_names = sorted(train_image_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])

    # In[ ]:

    predicted_batch = model.predict(train_image_batch)
    plot_predicted_batch(train_image_batch, train_label_batch, predicted_batch)

    # In[ ]:

    t = time.time()

    export_path = "/tmp/saved_models/{}".format(int(t))
    tf.keras.experimental.export_saved_model(model, export_path)

    print(export_path)

    # In[ ]:

    reloaded = tf.keras.experimental.load_from_saved_model(export_path,
                                                           custom_objects={'KerasLayer': hub.KerasLayer})

    result_batch = model.predict(train_image_batch)
    reloaded_result_batch = reloaded.predict(train_image_batch)

    print(abs(reloaded_result_batch - result_batch).max())

    # In[ ]:

    test_data_root = os.path.join(DATA_PATH, "test")
    test_image_data = read_data(test_data_root)

    test_loss, test_acc = model.evaluate(test_image_data)

    print('Test accuracy: {}'.format(test_acc))

    _, _, test_image_data = eval_model_on_data(model, "test")

    # In[ ]:

    test_image_batch, test_label_batch = test_image_data.next()
    predicted_batch = model.predict(test_image_batch)
    plot_predicted_batch(test_image_batch, test_label_batch, predicted_batch)

    # In[20]:
    if ENVIRONMENT == 'DGX':
        DATA_PATH = '/notebooks/userdata/Team A/mobilenet_data'

    #get_ipython().system('mkdir -p $DATA_PATH/test_boats/nature')
    get_ipython().system('ln -s ../test/boat $DATA_PATH/test_boats/')
    #get_ipython().system('mkdir -p $DATA_PATH/train_boats/nature')
    get_ipython().system('ln -s ../train/boat $DATA_PATH/train_boats/')

    # In[21]:

    _, test_boats_acc, test_boats_image_data = eval_model_on_data(model, "test_boats")

    # In[22]:

    _, train_boats_acc, train_boats_image_data = eval_model_on_data(model, "train_boats")

    # In[23]:

    print('Missed {} out of {} boats in test.zip'.format(
        int(test_boats_image_data.samples * (1 - test_boats_acc)),
        test_boats_image_data.samples))
    print('Missed {} out of {} boats in train.zip'.format(
        int(train_boats_image_data.samples * (1 - train_boats_acc)),
        train_boats_image_data.samples))

    # In[ ]:

if __name__ == '__main__':
    main()