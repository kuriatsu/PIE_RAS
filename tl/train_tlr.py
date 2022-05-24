#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import pandas as pd

def model():
    tlcat_model = Sequential()
    tlcat_model.add(BatchNormalization(input_shape=(32, 32, 3)))

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

    tlcat_model.add(Dense(3, activation="softmax"))
    tlcat_model.summary()
    tlcat_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return tlcat_model


def train(x_train, y_train, x_valid, y_valid):
    tlcat_model = model()
    checkpointer = ModelCheckpoint(filepath="model.weights.traffic_lights.hdf5", verbose=1, save_best_only=True)
    tlcat_model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)

def test(x_test, y_test, x_test_id):
    tlcat_model = model()
    tlcat_model.load_weights("model.weights.traffic_lights.hdf5")
    # predictions = [np.argmax(tlcat_model.predict(np.expand_dims(feature, axis=0))) for feature in x_test]
    df = pd.DataFrame(columns=["id", "result", "gt"])
    for i in range(0, len(x_test)):
        result = np.argmax(tlcat_model.predict(np.expand_dims(x_test[i], axis=0)))
        gt = np.argmax(y_test[i])
        buf = pd.DataFrame([(x_test_id[i], result, gt)], columns=df.columns)
        df = pd.concat([df, buf])
    # test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(y_test, axis=1))/len(predictions)
    print(df)
    df.to_csv("tlr_result.csv")
    # print('Test accuracy: %.4f%%' % test_accuracy)


def main():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    x_test_id = []
    y_test = []

    image_set_nums = {
        'train': ['set01', 'set02', 'set04'],
        'val'  : ['set05', 'set06'],
        'test' : ['test'],
        'all'  : ['set01', 'set02', 'set03',
                  'set04', 'set05', 'set06'],
        }
    with open("/home/kuriatsu/share/database.pickle", "rb") as f:
        database = pickle.load(f)

    for data in database:
        if data.get("set") in image_set_nums.get("train"):
            x_train.append(data.get("image"))
            y_train.append(data.get("state"))
        elif data.get("set") in image_set_nums.get("val"):
            x_valid.append(data.get("image"))
            y_valid.append(data.get("state"))
        elif data.get("set") in image_set_nums.get("test"):
            x_test.append(data.get("image"))
            x_test_id.append(data.get("id"))
            y_test.append(data.get("state"))

    print("len", len(np.array(y_test)), len(x_test[0]), len(x_test[0][0]))
    # train(np.array(x_train), np.array(y_train), np.array(x_valid), np.array(y_valid))
    test(x_test, y_test, x_test_id)

if __name__ == "__main__":
    main()
