import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# 整数を単語に戻す
def prepare_data():
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    # 単語を整数にマッピングする辞書
    word_index = imdb.get_word_index()

    # インデックスの最初の方は予約済み
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # モデルに入れるためのデータの準備
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)
    return (train_data, train_labels), (test_data, test_labels), reverse_word_index


def decode_review(reverse_word_index, text):
   return ' '.join([reverse_word_index.get(i, '?') for i in text])


def create_model():
    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels):
    # 検証用データを作る
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # モデルの訓練
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)
    return history


def show_loss_graph(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def show_acc_graph(history):
    plt.clf()   # 図のクリア
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label='Training acc')
    plt.plot(epochs, val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def main():
    (train_data, train_labels), (test_data, test_labels), reverse_word_index = prepare_data()
    # print(decode_review(reverse_word_index, train_data[0]))
    model = create_model()
    history = train_model(model, train_data, train_labels)
    # モデルの評価 [loss, acc]
    results = model.evaluate(test_data, test_labels)
    show_loss_graph(history)
    show_acc_graph(history)


if __name__ == '__main__':
    main()
