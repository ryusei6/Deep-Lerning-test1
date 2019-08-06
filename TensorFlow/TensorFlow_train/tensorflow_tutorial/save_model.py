from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras


def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    return model

def save_weight(model, train_images, train_labels, test_images, test_labels):
    checkpoint_path = os.path.splitext(__file__)[0]+'/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True,
            # 重みを5エポックごとに保存します
            period=5)

    model.fit(train_images, train_labels,
              epochs = 50, callbacks = [cp_callback],
              validation_data = (test_images,test_labels),
              verbose=0)

    latest_weights = tf.train.latest_checkpoint(checkpoint_dir)
    return latest_weights


def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = create_model()

    # 学習して重みを保存
    # latest_weights = save_weight(model, train_images, train_labels, test_images, test_labels)

    # 重みを読み込み
    latest_weights = os.path.splitext(__file__)[0]+'/cp-0050.ckpt'

    model.load_weights(latest_weights)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    main()
