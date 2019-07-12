import os, glob
import numpy as np
import random, math
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from keras.layers.core import Activation, Flatten
from keras.layers import BatchNormalization, Dropout



img_data = []
img_label = []
categories = ['LarryPage_face','JeffBezos_face']


def _make_sample(files):
    for category, file_name in files:
        _add_sample(category, file_name)
    return np.array(img_data), np.array(img_label)


def _add_sample(category, file_name):
    img = Image.open(file_name)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    data = np.asarray(img)
    img_data.append(data)
    img_label.append(category)


all_files = []
def read_img():
    root_dir = './data/imgs'
    for idx, category in enumerate(categories):
        img_dir = root_dir + '/' + category
        files = glob.glob(img_dir + '/*.jpg')
        for file in files:
            all_files.append([idx, file])
    random.shuffle(all_files)
    th = math.floor(len(all_files) * 0.8)
    train = all_files[0:th]
    test  = all_files[th:]
    x_train, y_train = _make_sample(train)
    x_test, y_test = _make_sample(test)

    n_classes = len(categories)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    return x_train, y_train, x_test, y_test
    # xy = (x_train, x_test, y_train, y_test)
    # print(x_test.shape)


def learn_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', activation='relu',input_shape=(150,150,3)))
    model.add(Conv2D(64, (3,3), padding='same',activation='relu'))
    model.add(MaxPool2D()) # model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='sgd',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train,
                      y_train,
                      epochs=10,
                      batch_size=6,
                      validation_data=(x_test,y_test))
    return model


def show_result(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def main():
    x_train, y_train, x_test, y_test = read_img()
    history = learn_model(x_train, y_train, x_test, y_test)
    show_result(history)


if __name__ == '__main__':
    main()
