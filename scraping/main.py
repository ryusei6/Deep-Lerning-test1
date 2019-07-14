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
categories = ['LarryPage_face','JeffBezos_face','MarkZuckerberg_face']
img_size = [150,150,3]
n_classes = len(categories)

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
    model.add(Conv2D(64, (3,3), padding='same', activation='relu',input_shape=(img_size[0],img_size[1],img_size[2])))
    model.add(Conv2D(64, (3,3), padding='same',activation='relu'))
    model.add(MaxPool2D()) # model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='sgd',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train,
                      y_train,
                      epochs=10,
                      batch_size=6,
                      validation_data=(x_test,y_test))
    return model, history


# グラフ
def plot_history_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


def plot_history_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


# imgサンプル
def img_show(model, x_train, y_train):
    predict = model.predict(x_train)
    plt.figure(figsize=(20,20))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.title('predict: '+str(predict[i].argmax())+'\nanswer: '+str(y_train[i].argmax()))
        plt.axis("off")
        plt.imshow(x_train[i].reshape(150,150,3))
    plt.show()


def main():
    x_train, y_train, x_test, y_test = read_img()
    model, history = learn_model(x_train, y_train, x_test, y_test)
    plot_history_loss(history)
    plot_history_acc(history)
    img_show(model, x_train, y_train)


if __name__ == '__main__':
    main()
