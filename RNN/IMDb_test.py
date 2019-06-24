from keras.models import Sequential
from keras.datasets import imdb
from keras.layers import Dense, Activation, Embedding, LSTM, GRU, SimpleRNN
from keras.optimizers import Adadelta
from keras.utils import np_utils, to_categorical
from keras.preprocessing import sequence
import numpy as np
from matplotlib import pyplot as plt

max_features = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)

max_len = 180
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

for k in range(10):
    # 0,1,2はそれぞれ「パディング」「シーケンス開始」「不明」のインデックスとして予約されている。
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_train[k]])
    if y_train[k] == 1:
        print('positive')
    else:
        print('negative')
    print(decoded_review)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)


# モデル
model = Sequential()
model.add(Embedding(input_dim=max_features,
                                  output_dim=128,
                                  input_length=max_len))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=5, validation_split=0.1)


# 検証
# positive
test_sentence = "It is a wonderful movie that I want to go to see again."
# negative
# test_sentence = "This movie is really terrible."

def split_text(text):
    result = []
    for word in text.split():
        word = word.lower()
        try:
            index = word_index[word]
        except(KeyError):
            index = 0
        result.append(index)
    return result

x = [split_text(test_sentence)]
x = sequence.pad_sequences(x, maxlen=max_len)
print(x)
y = model.predict(x)
print(y)
