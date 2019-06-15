import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

batch_size = 128
n_classes = 10
epochs = 1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from keras.layers.core import Activation, Flatten
from keras.layers import BatchNormalization, Dropout

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3,3), padding='same',activation='relu'))
model.add(MaxPool2D()) # model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(optimizer='sgd',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1, #途中経過を出力する(0: 出力しない 2:詳細に出す)
         validation_split=0.2) # 20%のデータを検証用に使う
score = model.evaluate(x_test, y_test)
print('print loss = ', score[0])
print('print accuracy = ', score[1])
