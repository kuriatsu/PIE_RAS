#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dence
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import pickle

def model():
    tlcat_model = Sequential()
    tlcat_model.add(BatchNormalization(imput_shape=(32, 32, 3)))

    tlcat_model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
    tlcat_model.add(MaxPooling2D(pool_size=2))
    tlcat_model.add(BatchNormalization())

    tlcat_model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
    tlcat_model.add(MaxPooling2D(pool_size=2))
    tlcat_model.add(BatchNormalization())

    tlcat_model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    tlcat_model.add(MaxPooling2D(pool_size=2))
    tlcat_model.add(BatchNormalization())

    tlcat_model.add(GlobalAveragePooling2D())

    tlcat_model.add(Dence(3, activation="softmax"))
    tlcat_model.summary()
    tlcat_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics="accuracy")
    return tlcat_model


def train(x_train, y_train, x_valid, x_valid):
    tlcat_model = model()
    checkpointer = ModelCheckpoint(filepath="model.weights.traffic_lights.hdf5", verbose=1, save_best_only=True)
    tlcat_model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)

def test(x_test, y_test):
    tlcat_model = model()
    tlcat_model.load_weights("model.weights.traffic_lights.hdf5")
    predictions = [np.argmax(tlcat_model.predict(np.expand_dims(feature, axis=0))) for feature in x_test]
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(y_test, axis=1))/len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


def main():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []

    image_set_nums = {
        'train': ['set01', 'set02', 'set04'],
        'val'  : ['set05', 'set06'],
        'test' : ['set03'],
        'all'  : ['set01', 'set02', 'set03',
                  'set04', 'set05', 'set06']
        }

    base_dir = "/media/kuriatsu/InternalHDD/PIE"
    for set in image_set_nums.get("train"):
        database = pickle.load(base_dir+"/tlr/"+set+".pickle")
        for data in database:
            x_train.append(data.get("image"))
            y_train.append(data.get("state"))

    for set in image_set_nums.get("val"):
        database = pickle.load(base_dir+"/tlr/"+set+".pickle")
        for data in database:
            x_valid.append(data.get("image"))
            y_valid.append(data.get("state"))

    train(x_train, y_train, x_valid, x_valid)
    test(x_test, y_test)
