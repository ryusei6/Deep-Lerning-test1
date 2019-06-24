from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation, Embedding, LSTM, GRU, SimpleRNN
from keras.optimizers import Adadelta
from keras.utils import np_utils, to_categorical
from keras.preprocessing import sequence
import numpy as np
from matplotlib import pyplot as plt
import string

#mnist
(X_train, y_train_orig), (X_test, y_test_orig) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_train /= 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
X_test /= 255.0
print(X_train.shape)


chars = [x for x in string.ascii_letters][:26]
chars.insert(0, '$')
print(chars)
labels = ['zero$', 'one$$', 'two$$', 'three', 'four$', 'five$', 'six$$', 'seven', 'eight', 'nine$']
# indexに変換
def convert_features(y):
    y_tmp = []
    for x in y:
        v = list(labels[x])
        v = [chars.index(x) for x in list(labels[x])]
        y_tmp.append(v)
    return y_tmp

maxlen=5
y_train = convert_features(y_train_orig)
y_train = sequence.pad_sequences(y_train, maxlen=maxlen)
y_train = to_categorical(y_train)

y_test = convert_features(y_test_orig)
y_test = sequence.pad_sequences(y_train, maxlen=maxlen)
y_test = to_categorical(y_test)


# モデル
from keras.layers import Dense, Conv2D, BatchNormalization
from keras.layers import Activation, Flatten, Dropout, UpSampling2D, MaxPooling2D, Reshape, GlobalAveragePooling2D
from keras.layers.core import RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

model = Sequential()
batch_size = 16

# Encoder
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(GlobalAveragePooling2D())

# Decoder
n_output = 26+1
n_hidden = 64
model.add(RepeatVector(maxlen))
model.add(LSTM(n_hidden,return_sequences=True,batch_size=batch_size))
model.add(TimeDistributed(Dense(n_output)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_split=0.1)


def vector_to_text(vec):
    text = ''
    for v in vec:
        if v != 0:
            print(chars[v])
            text += chars[v]
    return text

from matplotlib import cm
# テストデータ選択
test_index = 10
results = model.predict_classes(X_test)
result = results[test_index]
plt.imshow(X_test[test_index].reshape(28, 28), cmap=cm.gray_r)
plt.show()
print('predicted = ', vector_to_text(result))
